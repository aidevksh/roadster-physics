"""
로드스터 물리 교차 검증 스크립트 (Python 표준 라이브러리만 사용)

HTML 페이지의 JavaScript 엔진과 '따로' 만든 두 번째 구현입니다.
- JS 엔진: 반암시적 오일러 적분, dt = 0.5 ms
- 이 스크립트: 4차 룽게-쿠타(RK4) 적분, dt = 1 ms
두 구현의 결과가 1% 이내로 같으면, 계산 결과가 적분 방법이나 시간 간격에 기대지 않는다는 뜻입니다.

실행: python3 verify_physics.py
"""
import math

G, RHO = 9.81, 1.225
MPH, KMH, FT = 0.44704, 1 / 3.6, 0.3048
QUARTER = 402.336
R_N2, P_ATM = 296.8, 101325.0

# 기준 차량 (JS 엔진의 PRESETS 와 같은 가정값)
PERF = dict(m=1700, drive="RWD", mu=1.26, rear=0.55, h=0.40, L=2.825, torque=1400 * 8.4,
            r=0.367, P=1.0e6, CdA=0.50, Crr=0.012, ClA=1.0, fanG=1.0, fanV=22.35,
            gear=8.4, rpm=20500, T=0.0, ve=600.0, gas=0.0)
CARS = {
    "2017 프로토타입": dict(PERF, m=2000, drive="AWD", mu=1.30, rear=0.50, h=0.45, L=2.70,
                       torque=10000, r=0.35, P=0.85e6, ClA=0.3, fanG=0.0, gear=0, rpm=0),
    "2026 Performance": PERF,
    "2026 Thrust": dict(PERF, T=20000.0, gas=75.0, rpm=22500),
    "시연 0.81초 런": dict(PERF, T=30000.0, gas=75.0, rpm=22500),
    "2026 Rocket": dict(PERF, T=48000.0, gas=250.0, rpm=22500),
}


def fan_force(c, v):
    if not c["fanG"]:
        return 0.0
    f0, v1 = c["fanG"] * c["m"] * G, c["fanV"]
    if v <= v1:
        return f0
    return max(0.0, f0 * (2 * v1 - v) / v1)


def accel(c, v, gas):
    """속도 v, 남은 가스 gas 일 때의 가속도 [m/s²]"""
    m = c["m"] + gas
    thrust = c["T"] if gas > 0 else 0.0
    down = fan_force(c, v) + 0.5 * RHO * c["ClA"] * v * v
    n_total = m * G + down
    if c["drive"] == "AWD":
        grip = c["mu"] * n_total
    else:  # 후륜: F = μ·N_r / (1 − μh/L)
        grip = c["mu"] * c["rear"] * n_total / (1 - c["mu"] * c["h"] / c["L"])
    motor = min(c["torque"] / c["r"], c["P"] / v if v > 0.01 else math.inf)
    resist = 0.5 * RHO * c["CdA"] * v * v + c["Crr"] * n_total
    return (min(grip, motor) + thrust - resist) / m


def launch(c, x_end=QUARTER, dt=0.001):
    """RK4로 0→1/4마일 적분. 반환: 이벤트 시각 사전"""
    v_rpm = 2 * math.pi * c["r"] * c["rpm"] / 60 / c["gear"] if c["gear"] else math.inf
    t = x = v = 0.0
    gas = c["gas"]
    marks = {"t1ft": ("x", FT), "t60": ("v", 60 * MPH), "t100k": ("v", 100 * KMH),
             "t100mph": ("v", 100 * MPH), "tQuarter": ("x", QUARTER)}
    ev, v_end = {}, 0.0
    while x < x_end and t < 30:
        k1v, k1x = accel(c, v, gas), v
        k2v, k2x = accel(c, v + 0.5 * dt * k1v, gas), v + 0.5 * dt * k1v
        k3v, k3x = accel(c, v + 0.5 * dt * k2v, gas), v + 0.5 * dt * k2v
        k4v, k4x = accel(c, v + dt * k3v, gas), v + dt * k3v
        v_new = min(v + dt / 6 * (k1v + 2 * k2v + 2 * k3v + k4v), v_rpm)
        x_new = x + dt / 6 * (k1x + 2 * k2x + 2 * k3x + k4x)
        if gas > 0:
            gas = max(0.0, gas - c["T"] / c["ve"] * dt)
        for key, (kind, target) in marks.items():
            if key in ev:
                continue
            prev, cur = (x, x_new) if kind == "x" else (v, v_new)
            if cur >= target:
                ev[key] = t + dt * (target - prev) / (cur - prev)
                if key == "tQuarter":
                    v_end = v + (v_new - v) * (target - prev) / (cur - prev)
        t, x, v = t + dt, x_new, v_new
    ev["t60ro"] = ev["t60"] - ev["t1ft"]
    ev["vQuarter"] = v_end
    return ev


def rollout_accel(v, t_meas, d=FT):
    """롤아웃 차감 계측 시간 → 등가속도 a (이차방정식 근의 공식)"""
    b = math.sqrt(2 * d)
    x = (b + math.sqrt(b * b + 4 * v * t_meas)) / (2 * v)
    return 1 / (x * x)


def road_power(c, v):
    down = fan_force(c, v) + 0.5 * RHO * c["ClA"] * v * v
    return (0.5 * RHO * c["CdA"] * v * v + c["Crr"] * (c["m"] * G + down)) * v


def top_speed_power(c):
    lo, hi = 1.0, 400.0
    for _ in range(100):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if c["P"] > road_power(c, mid) else (lo, mid)
    return lo


def brake_distance(c, v0, dt=1e-4):
    """최대 제동(ABS 이상화) 거리: RK4 대신 아주 작은 dt의 오일러"""
    v = v0
    x = 0.0
    while v > 0:
        down = fan_force(c, v) + 0.5 * RHO * c["ClA"] * v * v
        n = c["m"] * G + down
        a = (c["mu"] * n + 0.5 * RHO * c["CdA"] * v * v + c["Crr"] * n) / c["m"]
        v_new = max(0.0, v - a * dt)
        x += 0.5 * (v + v_new) * (dt if v_new > 0 else v / a)
        v = v_new
    return x


def n2_z(p, T):
    """Peng–Robinson 상태방정식으로 질소의 압축인자 Z (이분법으로 독립 풀이)"""
    tc, pc, w, ru = 126.2, 3.3958e6, 0.0372, 8.314
    kappa = 0.37464 + 1.54226 * w - 0.26992 * w * w
    alpha = (1 + kappa * (1 - math.sqrt(T / tc))) ** 2
    a = 0.45724 * ru * ru * tc * tc / pc * alpha
    b = 0.0778 * ru * tc / pc
    A, B = a * p / (ru * T) ** 2, b * p / (ru * T)
    f = lambda z: z ** 3 - (1 - B) * z ** 2 + (A - 3 * B * B - 2 * B) * z - (A * B - B * B - B ** 3)
    lo, hi = B + 1e-9, 5.0
    for _ in range(200):
        mid = (lo + hi) / 2
        lo, hi = (lo, mid) if f(lo) * f(mid) <= 0 else (mid, hi)
    return (lo + hi) / 2


def gas_mass(liters, bar, T=293.0):
    p = bar * 1e5
    return p * liters / 1000 / (n2_z(p, T) * R_N2 * T)


def line(title):
    print(f"\n── {title} " + "─" * max(0, 52 - len(title)))


if __name__ == "__main__":
    line("1. 직선 가속 (RK4, dt = 1 ms)")
    for name, car in CARS.items():
        e = launch(car)
        print(f"{name:<17} 0-60 mph {e['t60']:.2f} s (롤아웃 {e['t60ro']:.2f} s) | "
              f"0-100 km/h {e['t100k']:.2f} s | 0-100 mph {e['t100mph']:.2f} s | "
              f"1/4 mile {e['tQuarter']:.2f} s @ {e['vQuarter'] * 3.6:.0f} km/h")

    line("2. 발표 시간 → 필요한 평균 가속도")
    v60 = 60 * MPH
    for label, t in [("2017 0-60 1.9 s", 1.9), ("Performance 1.6 s", 1.6),
                     ("Thrust 0.85 s", 0.85), ("Rocket 0.55 s", 0.55)]:
        print(f"{label:<20} 단순 a = {v60 / t / G:.2f} g | 롤아웃 반영 a = {rollout_accel(v60, t) / G:.2f} g")
    a081 = v60 / 0.81
    print(f"시연 0.81 s → 평균 {a081 / G:.2f} g, 이동거리 {0.5 * v60 * 0.81:.1f} m, "
          f"롤아웃 차감 {0.81 - math.sqrt(2 * FT / a081):.3f} s (테슬라 발표 0.67 s)")

    line("3. 최고속도와 일률")
    c17, perf = CARS["2017 프로토타입"], CARS["2026 Performance"]
    p400 = road_power(c17, 400 * KMH)
    print(f"2017 · 400 km/h 유지 일률 {p400 / 1e3:.0f} kW ({p400 / 735.5:.0f} PS)")
    vp = top_speed_power(perf)
    v_rpm = lambda rpm: 2 * math.pi * perf["r"] * rpm / 60 / perf["gear"]
    print(f"2026 · 1 MW 출력 한계 최고속도 {vp * 3.6:.0f} km/h ({vp / MPH:.0f} mph)")
    print(f"2026 · 회전수 한계 20,500 rpm → {v_rpm(20500) / MPH:.0f} mph, 22,500 rpm → {v_rpm(22500) / MPH:.0f} mph")
    p_bat = p400 / 0.90
    print(f"2017 · 200 kWh로 400 km/h 주행: {200 / (p_bat / 1e3) * 60:.1f} 분, "
          f"{400 * 200 / (p_bat / 1e3):.0f} km, C-rate {p_bat / 200e3:.2f}")

    line("4. 코너링")
    for r in (30, 100, 300):
        v_t = math.sqrt(perf["mu"] * G * r)
        print(f"반지름 {r:>3} m · 타이어만 최고 {v_t * 3.6:5.1f} km/h")
    print(f"전복 한계 w/(2h): 로드스터 {1.70 / 0.80:.2f} g, SUV(h 0.70 m, w 1.62 m) {1.62 / 1.40:.2f} g")
    print(f"팬 1 g 흡입 시 횡가속 한계 μ(1 + 1) = {perf['mu'] * 2:.2f} g")

    line("5. 제동 (100 → 0 km/h)")
    v100 = 100 * KMH
    for mu in (1.0, 1.2):
        print(f"이론식 v²/(2μg), μ = {mu}: {v100 ** 2 / (2 * mu * G):.1f} m")
    no_fan = dict(perf, fanG=0.0)
    print(f"2026 · 팬 없음 {brake_distance(no_fan, v100):.1f} m | 팬 1 g {brake_distance(perf, v100):.1f} m")
    ek = 0.5 * perf["m"] * v100 ** 2
    print(f"운동에너지 ½mv² = {ek / 1e3:.0f} kJ · 400 km/h(2,000 kg)에서는 {0.5 * 2000 * (400 * KMH) ** 2 / 1e6:.2f} MJ")

    line("6. 냉가스 추진기 (질소, 293 K)")
    for liters, bar in [(200, 380), (200, 690), (600, 380), (600, 450)]:
        z = n2_z(bar * 1e5, 293.0)
        print(f"{liters} L · {bar} bar → Z = {z:.3f}, 질소 {gas_mass(liters, bar):.1f} kg")
    for T, gas in [(80e3, 75), (80e3, 250)]:
        mdot = T / 600
        dv = 600 * math.log((1700 + gas) / 1700)
        hover = 600 / G * math.log((1700 + gas) / 1700)
        print(f"추력 {T / 1e3:.0f} kN · 가스 {gas} kg → ṁ {mdot:.0f} kg/s, 최대추력 {gas / mdot:.2f} s, "
              f"충격량 {gas * 600 / 1e3:.0f} kN·s, Δv {dv:.1f} m/s, 호버 {hover:.1f} s")
    mu = perf["mu"]
    print(f"최적 분사각 tanθ = μ → θ = {math.degrees(math.atan(mu)):.1f}°, 효과 배율 √(1+μ²) = {math.hypot(1, mu):.3f}")
    w_iso = 250 * R_N2 * 293 * math.log(380e5 / P_ATM)
    print(f"250 kg을 380 bar로 다시 채우는 최소 일(등온) {w_iso / 3.6e6:.1f} kWh")

    line("7. 시연 물리")
    R = 10.5
    print(f"루프 반지름 {R} m · 흡입 없음: 꼭대기 최소 {math.sqrt(G * R):.2f} m/s, 바닥 최소 {math.sqrt(5 * G * R):.2f} m/s")
    print(f"4,700 kg 상당 흡입 · 정지 상태로 거꾸로: N = F_s − mg = {(4700 - 1700) * G / 1e3:.1f} kN")
    vy = math.sqrt(2 * G * 7)
    print(f"7 m 높이 → 필요한 수직 속도 {vy:.2f} m/s, 체공 {2 * vy / G:.2f} s")
    print(f"폭 400 m 협곡 · 45° 발사 최소 속도(공기저항 무시) {math.sqrt(400 * G):.1f} m/s = {math.sqrt(400 * G) * 3.6:.0f} km/h")
