! Verifica el RK4 en Fortran contra la solucion analitica de la caida con arrastre
! cuadratico y comprueba su orden de convergencia (AC-5, AC-6, AC-9 de la spec 001).
program verify
  use caustix_rk4, only: wp, simulate, status_ok
  implicit none

  real(wp), parameter :: g = 9.80665_wp, mass = 1.0_wp
  real(wp), parameter :: rho = 1.225_wp, cd = 0.47_wp, area = 0.01_wp
  real(wp) :: k, v_t, y0
  logical :: ok

  k = rho * cd * area / (2.0_wp * mass)
  v_t = sqrt(g / k)
  y0 = 10000.0_wp
  ok = .true.

  call check_precision()
  call check_terminal()
  call check_order()
  call check_impact_order()

  if (.not. ok) error stop 'verify: FALLO'
  print '(A)', 'verify: OK (precision, terminal, orden RK4, orden del impacto)'

contains

  function analytic_y(t, start) result(y)
    real(wp), intent(in) :: t, start
    real(wp) :: y
    y = start - (v_t**2 / g) * log(cosh(g * t / v_t))
  end function analytic_y

  function analytic_v(t) result(v)
    real(wp), intent(in) :: t
    real(wp) :: v
    v = -v_t * tanh(g * t / v_t)
  end function analytic_v

  subroutine run_fall(start, dt, t_max, with_ground, n, t, pos, vel, hit)
    real(wp), intent(in) :: start, dt, t_max
    logical, intent(in) :: with_ground
    integer, intent(out) :: n
    real(wp), allocatable, intent(out) :: t(:), pos(:, :), vel(:, :)
    logical, intent(out) :: hit
    real(wp) :: apex_t, apex_z, ground
    integer :: status
    ground = 0.0_wp
    if (.not. with_ground) ground = -1.0e30_wp
    call simulate(2, k, g, ground, dt, t_max, 100000000, [0.0_wp, start], [0.0_wp, 0.0_wp], &
                  n, t, pos, vel, hit, apex_t, apex_z, status)
    if (status /= status_ok) error stop 'verify: la simulacion fallo'
  end subroutine run_fall

  subroutine check_precision()
    integer :: n, i
    real(wp), allocatable :: t(:), pos(:, :), vel(:, :)
    logical :: hit
    real(wp) :: worst_y, worst_v
    call run_fall(y0, 1.0e-3_wp, 5.0_wp, .false., n, t, pos, vel, hit)
    worst_y = 0.0_wp
    worst_v = 0.0_wp
    do i = 2, n
      worst_y = max(worst_y, abs(pos(2, i) - analytic_y(t(i), y0)) / abs(analytic_y(t(i), y0)))
      worst_v = max(worst_v, abs(vel(2, i) - analytic_v(t(i))) / abs(analytic_v(t(i))))
    end do
    print '(A,2ES12.3)', 'precision (error relativo y, v): ', worst_y, worst_v
    if (worst_y > 1.0e-8_wp .or. worst_v > 1.0e-8_wp) ok = .false.
  end subroutine check_precision

  subroutine check_terminal()
    integer :: n
    real(wp), allocatable :: t(:), pos(:, :), vel(:, :)
    logical :: hit
    call run_fall(y0, 0.05_wp, 70.0_wp, .false., n, t, pos, vel, hit)
    print '(A,ES12.3)', 'terminal |v|/v_t - 1: ', abs(abs(vel(2, n)) / v_t - 1.0_wp)
    if (abs(abs(vel(2, n)) / v_t - 1.0_wp) > 1.0e-3_wp) ok = .false.
  end subroutine check_terminal

  subroutine check_order()
    real(wp), parameter :: dts(4) = [0.4_wp, 0.2_wp, 0.1_wp, 0.05_wp]
    real(wp) :: err_y(4), err_v(4), order
    integer :: i, n
    real(wp), allocatable :: t(:), pos(:, :), vel(:, :)
    logical :: hit
    do i = 1, 4
      call run_fall(y0, dts(i), 16.0_wp, .false., n, t, pos, vel, hit)
      err_y(i) = abs(pos(2, n) - analytic_y(t(n), y0))
      err_v(i) = abs(vel(2, n) - analytic_v(t(n)))
    end do
    do i = 1, 3
      order = log(err_y(i) / err_y(i + 1)) / log(2.0_wp)
      print '(A,I0,A,F8.3)', 'orden y (par ', i, '): ', order
      if (order < 3.8_wp .or. order > 4.2_wp) ok = .false.
      order = log(err_v(i) / err_v(i + 1)) / log(2.0_wp)
      print '(A,I0,A,F8.3)', 'orden v (par ', i, '): ', order
      if (order < 3.8_wp .or. order > 4.2_wp) ok = .false.
    end do
  end subroutine check_order

  subroutine check_impact_order()
    real(wp), parameter :: dts(4) = [0.4_wp, 0.2_wp, 0.1_wp, 0.05_wp]
    real(wp) :: start, t_star, err(4), order
    integer :: i, n
    real(wp), allocatable :: t(:), pos(:, :), vel(:, :)
    logical :: hit
    start = 100.0_wp
    t_star = (v_t / g) * acosh(exp(g * start / v_t**2))
    do i = 1, 4
      call run_fall(start, dts(i), 60.0_wp, .true., n, t, pos, vel, hit)
      if (.not. hit) ok = .false.
      err(i) = abs(t(n) - t_star)
    end do
    do i = 1, 3
      order = log(err(i) / err(i + 1)) / log(2.0_wp)
      print '(A,I0,A,F8.3)', 'orden del impacto (par ', i, '): ', order
      if (order < 3.0_wp) ok = .false.
    end do
  end subroutine check_impact_order

end program verify
