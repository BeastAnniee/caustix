# Changelog

Formato basado en [Keep a Changelog](https://keepachangelog.com/es-ES/1.1.0/) y versionado
[SemVer](https://semver.org/lang/es/).

## [1.0.0] - sin publicar

### Añadido
- `mru` y `mrua`: cinemática analítica en 1 a 3 dimensiones.
- `projectile`: tiro parabólico sin arrastre por solución analítica, con suelo configurable.
- `projectile_with_drag`: tiro con arrastre cuadrático (RK4 de paso fijo) con impacto y ápice
  localizados por interpolación de Hermite.
- `launch_velocity`, `Motion`, `Trajectory` (energía cinética, potencial y mecánica por muestra),
  `IntegrationError` y `plot_trajectory` (extra `caustix[plot]`).
- Validador independiente en Fortran (solo desarrollo, no publicado) y fixtures de contraste.
