! Genera la fixture CSV a partir de cases.txt.
! Uso: gen_fixtures <cases.txt> <salida.csv>
! Formato: lineas '#case,...' (parametros y resultados del caso) seguidas de filas
! 'name,i,t,x1,x2,x3,v1,v2,v3' (las componentes que no existen en 2D valen 0).
program gen_fixtures
  use caustix_rk4, only: wp, simulate, status_ok
  implicit none

  character(len=512) :: cases_path, out_path, line
  character(len=64) :: name
  integer :: unit_in, unit_out, ios, dim, stride, n, status, i
  real(wp) :: mass, rho, cd, area, g, ground, dt, t_max, k, apex_t, apex_z
  real(wp) :: p0(3), v0(3), pad_p(3), pad_v(3)
  real(wp), allocatable :: t(:), pos(:, :), vel(:, :)
  logical :: hit

  if (command_argument_count() /= 2) error stop 'uso: gen_fixtures cases.txt salida.csv'
  call get_command_argument(1, cases_path)
  call get_command_argument(2, out_path)

  open (newunit=unit_in, file=trim(cases_path), status='old', action='read', iostat=ios)
  if (ios /= 0) error stop 'no se puede abrir cases.txt'
  open (newunit=unit_out, file=trim(out_path), status='replace', action='write', iostat=ios)
  if (ios /= 0) error stop 'no se puede escribir la salida'

  write (unit_out, '(A)') '# Generado por fortran/tools/regen.py (gfortran, RK4 independiente). No editar.'

  do
    read (unit_in, '(A)', iostat=ios) line
    if (ios /= 0) exit
    if (len_trim(line) == 0 .or. line(1:1) == '#') cycle

    p0 = 0.0_wp
    v0 = 0.0_wp
    read (line, *) name, dim
    if (dim < 2 .or. dim > 3) error stop 'dim debe ser 2 o 3'
    call read_case(line, dim, name, mass, rho, cd, area, g, ground, dt, t_max, stride, p0, v0)

    k = rho * cd * area / (2.0_wp * mass)
    call simulate(dim, k, g, ground, dt, t_max, 1000000, p0(1:dim), v0(1:dim), &
                  n, t, pos, vel, hit, apex_t, apex_z, status)
    if (status /= status_ok) error stop 'la simulacion fallo'

    write (unit_out, '(A)') '#case,' // trim(name) // ',' // int_str(dim) // ',' // num(mass) // ',' &
      // num(rho) // ',' // num(cd) // ',' // num(area) // ',' // num(g) // ',' // num(ground) // ',' &
      // num(dt) // ',' // num(t_max) // ',' // int_str(stride) // ',' // int_str(merge(1, 0, hit)) &
      // ',' // num(t(n)) // ',' // num(apex_t) // ',' // num(apex_z - ground) // ',' // int_str(n) &
      // ',' // num(p0(1)) // ',' // num(p0(2)) // ',' // num(p0(3)) &
      // ',' // num(v0(1)) // ',' // num(v0(2)) // ',' // num(v0(3))

    do i = 1, n
      if (mod(i - 1, stride) /= 0 .and. i /= n) cycle
      pad_p = 0.0_wp
      pad_v = 0.0_wp
      pad_p(1:dim) = pos(:, i)
      pad_v(1:dim) = vel(:, i)
      write (unit_out, '(A)') trim(name) // ',' // int_str(i - 1) // ',' // num(t(i)) // ',' &
        // num(pad_p(1)) // ',' // num(pad_p(2)) // ',' // num(pad_p(3)) // ',' &
        // num(pad_v(1)) // ',' // num(pad_v(2)) // ',' // num(pad_v(3))
    end do
  end do

  close (unit_in)
  close (unit_out)

contains

  subroutine read_case(text, d, case_name, m, r, c, a, grav, floor, step, horizon, every, p, v)
    character(len=*), intent(in) :: text
    integer, intent(in) :: d
    character(len=*), intent(out) :: case_name
    real(wp), intent(out) :: m, r, c, a, grav, floor, step, horizon
    integer, intent(out) :: every
    real(wp), intent(inout) :: p(3), v(3)
    integer :: dd
    read (text, *) case_name, dd, m, r, c, a, grav, floor, step, horizon, every, p(1:d), v(1:d)
  end subroutine read_case

  function num(x) result(s)
    real(wp), intent(in) :: x
    character(len=:), allocatable :: s
    character(len=32) :: buffer
    write (buffer, '(ES25.16E3)') x
    s = trim(adjustl(buffer))
  end function num

  function int_str(m) result(s)
    integer, intent(in) :: m
    character(len=:), allocatable :: s
    character(len=16) :: buffer
    write (buffer, '(I0)') m
    s = trim(buffer)
  end function int_str

end program gen_fixtures
