! Validador independiente (solo desarrollo) del RK4 con arrastre cuadrático de caustix.
! Mismo algoritmo que python/src/caustix/{_rk4,_events}.py, escrito de forma independiente:
! RK4 de paso fijo, paradas en el primer paso bajo el suelo, impacto y apice por
! interpolacion cubica de Hermite con biseccion acotada a 60 iteraciones.
module caustix_rk4
  implicit none
  private

  integer, parameter, public :: wp = selected_real_kind(15, 307)
  integer, parameter :: bisection_iterations = 60
  integer, parameter :: scan_points = 2000
  real(wp), parameter :: step_count_tol = 1.0e-9_wp

  integer, parameter, public :: status_ok = 0
  integer, parameter, public :: status_max_steps = 1
  integer, parameter, public :: status_non_finite = 2

  public :: simulate, step_count

contains

  pure function accel(dim, k, g, v) result(a)
    integer, intent(in) :: dim
    real(wp), intent(in) :: k, g, v(dim)
    real(wp) :: a(dim)
    a = -k * sqrt(sum(v * v)) * v
    a(dim) = a(dim) - g
  end function accel

  pure function step_count(dt, t_max) result(n)
    real(wp), intent(in) :: dt, t_max
    integer :: n
    real(wp) :: ratio, nearest
    ratio = t_max / dt
    nearest = anint(ratio)
    if (abs(ratio - nearest) <= step_count_tol * max(1.0_wp, ratio)) then
      n = max(int(nearest), 1)
    else
      n = int(ceiling(ratio))
    end if
  end function step_count

  pure function hermite(dim, p0, m0, p1, m1, h, s) result(p)
    integer, intent(in) :: dim
    real(wp), intent(in) :: p0(dim), m0(dim), p1(dim), m1(dim), h, s
    real(wp) :: p(dim)
    real(wp) :: s2, s3, h00, h10, h01, h11
    s2 = s * s
    s3 = s2 * s
    h00 = 2.0_wp * s3 - 3.0_wp * s2 + 1.0_wp
    h10 = s3 - 2.0_wp * s2 + s
    h01 = -2.0_wp * s3 + 3.0_wp * s2
    h11 = s3 - s2
    p = h00 * p0 + h10 * h * m0 + h01 * p1 + h11 * h * m1
  end function hermite

  ! Raiz s en [0,1] de la componente vertical (ultima) del Hermite menos `level`;
  ! f(0) y f(1) deben tener signo opuesto.
  pure function root_s(dim, p0, m0, p1, m1, h, level) result(s_star)
    integer, intent(in) :: dim
    real(wp), intent(in) :: p0(dim), m0(dim), p1(dim), m1(dim), h, level
    real(wp) :: s_star
    real(wp) :: lo, hi, mid, f_mid(dim), f_best
    logical :: positive_lo
    integer :: iter
    lo = 0.0_wp
    hi = 1.0_wp
    ! Si el paso parte exactamente del nivel (f(0) = 0) hay una raiz espuria en s = 0:
    ! se acota el tramo al maximo de la cubica (barrido denso; es un validador).
    if ((p0(dim) - level) <= 0.0_wp) then
      f_best = p0(dim) - level
      do iter = 1, scan_points
        mid = real(iter, wp) / real(scan_points + 1, wp)
        f_mid = hermite(dim, p0, m0, p1, m1, h, mid)
        if ((f_mid(dim) - level) > f_best) then
          f_best = f_mid(dim) - level
          lo = mid
        end if
      end do
    end if
    f_mid = hermite(dim, p0, m0, p1, m1, h, lo)
    positive_lo = (f_mid(dim) - level) > 0.0_wp
    do iter = 1, bisection_iterations
      mid = 0.5_wp * (lo + hi)
      f_mid = hermite(dim, p0, m0, p1, m1, h, mid)
      if (((f_mid(dim) - level) > 0.0_wp) .eqv. positive_lo) then
        lo = mid
      else
        hi = mid
      end if
    end do
    s_star = 0.5_wp * (lo + hi)
  end function root_s

  ! Simula el tiro con arrastre. Salidas: n muestras (t, pos, vel con forma (dim, n)),
  ! `hit` (impacto con el suelo), instante y altura del apice (coordenada vertical).
  subroutine simulate(dim, k, g, ground, dt, t_max, max_steps, p0, v0, &
                      n, t, pos, vel, hit, apex_t, apex_z, status)
    integer, intent(in) :: dim, max_steps
    real(wp), intent(in) :: k, g, ground, dt, t_max, p0(dim), v0(dim)
    integer, intent(out) :: n, status
    real(wp), allocatable, intent(out) :: t(:), pos(:, :), vel(:, :)
    logical, intent(out) :: hit
    real(wp), intent(out) :: apex_t, apex_z

    integer :: nsteps, i, last, j
    real(wp) :: x(dim), v(dim), h, t_next
    real(wp) :: k1v(dim), k2v(dim), k3v(dim), k4v(dim), k2x(dim), k3x(dim), k4x(dim)
    real(wp) :: a0(dim), a1(dim), s_star, hh, px(dim)
    real(wp), allocatable :: tr(:), pr(:, :), vr(:, :)
    real(wp) :: z_scale, landing_tol
    logical :: crossed, found, on_ground

    status = status_ok
    n = 0
    hit = .false.
    apex_t = 0.0_wp
    apex_z = p0(dim)
    nsteps = step_count(dt, t_max)
    if (nsteps > max_steps) then
      status = status_max_steps
      allocate (t(0), pos(dim, 0), vel(dim, 0))
      return
    end if

    allocate (tr(0:nsteps), pr(dim, 0:nsteps), vr(dim, 0:nsteps))
    tr(0) = 0.0_wp
    pr(:, 0) = p0
    vr(:, 0) = v0
    x = p0
    v = v0
    crossed = .false.
    last = 0
    ! Residuo de redondeo admitido para el aterrizaje sobre una muestra (como _rk4.integrate)
    z_scale = max(1.0_wp, abs(p0(dim)), abs(ground))
    landing_tol = 4.0_wp * epsilon(1.0_wp) * real(max(nsteps, 16), wp)
    do i = 0, nsteps - 1
      if (i == nsteps - 1) then
        t_next = t_max
        h = t_max - real(i, wp) * dt
      else
        t_next = real(i + 1, wp) * dt
        h = dt
      end if
      k1v = accel(dim, k, g, v)
      k2v = accel(dim, k, g, v + 0.5_wp * h * k1v)
      k3v = accel(dim, k, g, v + 0.5_wp * h * k2v)
      k4v = accel(dim, k, g, v + h * k3v)
      k2x = v + 0.5_wp * h * k1v
      k3x = v + 0.5_wp * h * k2v
      k4x = v + h * k3v
      x = x + (h / 6.0_wp) * (v + 2.0_wp * k2x + 2.0_wp * k3x + k4x)
      v = v + (h / 6.0_wp) * (k1v + 2.0_wp * k2v + 2.0_wp * k3v + k4v)
      if (.not. (all(abs(x) <= huge(1.0_wp)) .and. all(abs(v) <= huge(1.0_wp)))) then
        status = status_non_finite
        allocate (t(0), pos(dim, 0), vel(dim, 0))
        return
      end if
      tr(i + 1) = t_next
      pr(:, i + 1) = x
      vr(:, i + 1) = v
      last = i + 1
      ! Cruce: bajo el suelo, o exactamente en el con velocidad vertical negativa
      ! (aterrizaje exacto sobre la muestra).
      z_scale = max(z_scale, abs(x(dim)))
      on_ground = abs(x(dim) - ground) <= landing_tol * z_scale .and. v(dim) < 0.0_wp
      if (x(dim) < ground .or. on_ground) then
        if (on_ground) pr(dim, i + 1) = ground
        crossed = .true.
        exit
      end if
    end do

    ! Apice sobre las muestras sin refinar (incluida la que cruza el suelo)
    if (vr(dim, 0) <= 0.0_wp) then
      apex_t = tr(0)
      apex_z = pr(dim, 0)
    else
      found = .false.
      do j = 0, last - 1
        if (vr(dim, j + 1) <= 0.0_wp) then
          found = .true.
          exit
        end if
      end do
      if (.not. found) then
        apex_t = tr(last)
        apex_z = pr(dim, last)
      else
        hh = tr(j + 1) - tr(j)
        a0 = accel(dim, k, g, vr(:, j))
        a1 = accel(dim, k, g, vr(:, j + 1))
        s_star = root_s(dim, vr(:, j), a0, vr(:, j + 1), a1, hh, 0.0_wp)
        apex_t = tr(j) + s_star * hh
        px = hermite(dim, pr(:, j), vr(:, j), pr(:, j + 1), vr(:, j + 1), hh, s_star)
        apex_z = px(dim)
      end if
    end if

    n = last + 1
    allocate (t(n), pos(dim, n), vel(dim, n))
    t = tr(0:last)
    pos = pr(:, 0:last)
    vel = vr(:, 0:last)
    hit = crossed
    if (crossed) then
      hh = tr(last) - tr(last - 1)
      a1 = accel(dim, k, g, vr(:, last))
      a0 = accel(dim, k, g, vr(:, last - 1))
      if (pr(dim, last) - ground == 0.0_wp) then
        s_star = 1.0_wp
        t(n) = tr(last)
      else
        s_star = root_s(dim, pr(:, last - 1), vr(:, last - 1), pr(:, last), vr(:, last), hh, ground)
        t(n) = tr(last - 1) + s_star * hh
      end if
      pos(:, n) = hermite(dim, pr(:, last - 1), vr(:, last - 1), pr(:, last), vr(:, last), hh, s_star)
      pos(dim, n) = ground
      vel(:, n) = hermite(dim, vr(:, last - 1), a0, vr(:, last), a1, hh, s_star)
    end if
  end subroutine simulate

end module caustix_rk4
