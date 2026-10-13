## Qué y por qué
<!-- 2-3 líneas. -->

**Spec:** `docs/specs/NNN-slug/spec.md` · **Riesgo:** R? · **Componentes:** …

## Evidencia
- Gate: `node scripts/tc-verify.mjs --all` → **PASS** (commit `<sha>`, salida en `validation.md`)
- Validación por criterio: `docs/specs/NNN-slug/validation.md`
- Review independiente: `docs/specs/NNN-slug/review.md` → sin BLOCKER/MAJOR abiertos

## Gates manuales pendientes (R3 / PLC / planta)
- [ ] Simulación / FAT con evidencia adjunta
- [ ] Segundo revisor con conocimiento del proceso
- [ ] Plan de puesta en marcha y rollback aprobado

## Despliegue y vuelta atrás
<!-- Cómo se despliega, cómo se revierte, quién ejecuta en entorno real. -->

## Revisores requeridos por riesgo
R1: 1 · R2: 1 + review de IA · R3: 2 (uno del proceso)

## Checklist del autor
- [ ] Sin secretos ni TODO/debug residuales · [ ] Docs/ADR al día · [ ] Divergencias del plan anotadas
