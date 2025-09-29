# Software Academia

Boilerplate inicial para el sistema de gestión de la academia de inglés.



Base de Datos Ingles_academia


1. Resumen

Base de datos del sistema CRM para la academia de inglés en Florida.
Permite gestionar estudiantes, pagos, asistencia, profesores y comunicación con alumnos.
Incluye soporte para dashboard, automatización de recordatorios y futuras integraciones de predicción de abandono (churn prediction).

2. Tablas principales

roles → catálogo de roles de usuario (ADMIN, STAFF, TEACHER).

users → usuarios internos (administrativos, profesores) con referencia a roles.

students → alumnos con datos de contacto.

teachers → docentes registrados.

courses → cursos/grupos ofrecidos por la academia.

course_teachers → asigna profesores a cursos + horarios.

enrollments → matrículas de alumnos en cursos, con fechas de inicio/fin y estado activo.

sessions → sesiones de clases programadas para cada curso.

attendance → registro de asistencia de estudiantes en cada sesión.

payments → pagos de los estudiantes (monto, fecha, período).

message_log → registro de correos/sms/whatsapp enviados a estudiantes.

3. Relaciones (claves foráneas)

users.role_id → roles.id

course_teachers.course_id → courses.id

course_teachers.teacher_id → teachers.Teacher_ID

enrollments.student_id → students.Student_id

enrollments.course_id → courses.id

sessions.course_id → courses.id

attendance.session_id → sessions.id

payments.student_id → students.Student_id

message_log.student_id → students.Student_id

4. Datos iniciales

Precargados (CSV):

students (students.csv)

teachers (teachers.csv)

payments (payments.csv)

attendance (attendance.csv)

message_log (emails.csv)

Inicializados por sistema:

roles: ADMIN, STAFF, TEACHER.

users: 1 usuario admin de prueba (admin@academia.com).

Vacías al inicio (se llenan con el uso del sistema):

courses, course_teachers, enrollments, sessions.

5. Índices importantes

students(email) → búsquedas rápidas y evitar duplicados.

enrollments(student_id, is_active) → listar alumnos activos.

sessions(course_id, class_date) → calendario de curso.

attendance(session_id) → control de asistencia por clase.

payments(student_id, period_month) → ingresos y morosidad.

6. Seguridad

Usuarios de BD:

admin_user: todos los permisos sobre Ingles_academia (solo para administración).

app_user: solo SELECT, INSERT, UPDATE, DELETE (usado por el backend).

Buenas prácticas:

Guardar credenciales en variables de entorno, no en código.

Limitar app_user al host/IP del servidor backend.

Usar TLS/SSL si la app y MySQL no están en la misma máquina.
