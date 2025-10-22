// Datos simulados
window.__DATA__ = {
  kpis: {
    highRiskCount: 18,
    monthlyRevenue: 5800
  },
  drivers: [
    { title: 'Low Attendance', value: 0.70, severity: 'warn' },
    { title: 'Low Grade', value: 0.49, severity: 'warn' },
    { title: 'Lack of Interaction', value: 0.30, severity: 'danger' },
    { title: 'Late Payments', value: 0.22, severity: 'danger' }
  ],
  progress: [
    { label: 'Student (0-100)', red: 32, blue: 82 },
    { label: 'Cohort A — Score', red: 18, blue: 60 }
  ],
  students: [
    {
      name: 'Ana Rodríguez',
      email: 'ana.rod@example.com',
      reasons: 'Low Attendance',
      risk: 'high',
      payment: 92,
      lastPayment: 'Dec 15',
      nextEvent: '15/01',
      status: 'At risk'
    },
    {
      name: 'Juan Pérez',
      email: 'juan.perez@example.com',
      reasons: 'Overdue Payment',
      risk: 'medium',
      payment: 41,
      lastPayment: 'Dec 03',
      nextEvent: '28/01',
      status: 'Overdue'
    },
    {
      name: 'Carlos Sánchez',
      email: 'carlos.sanchez@example.com',
      reasons: 'Low Grades',
      risk: 'high',
      payment: 90,
      lastPayment: 'Nov 22',
      nextEvent: '10/02',
      status: 'At risk'
    },
    {
      name: 'Rita Rentry',
      email: 'rita.rentry@example.com',
      reasons: 'Lack of Interaction',
      risk: 'low',
      payment: 246,
      lastPayment: 'Oct 28',
      nextEvent: '21/02',
      status: 'OK'
    }
  ],
  planTemplate: [
    'Personalizar SMS',
    'Enviar Email personalizado',
    'Ofrecer descuento / plan flexible',
    'Agendar llamada',
    'Seguimiento en X días'
  ]
};
