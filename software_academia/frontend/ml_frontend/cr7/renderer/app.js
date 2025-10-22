(() => {
  const d = window.document;
  const data = window.__DATA__;

  // KPIs
  d.getElementById('kpiHighRisk').textContent = data.kpis.highRiskCount;
  d.getElementById('kpiRevenue').textContent =
    `$${Number(data.kpis.monthlyRevenue).toLocaleString()}`;

  // Drivers
  const driversEl = d.getElementById('driversList');
  data.drivers.forEach(dr => {
    const wrap = d.createElement('div');
    wrap.className = 'driver';
    wrap.innerHTML = `
      <div class="title">⚠️ ${dr.title} <span style="margin-left:auto; color:#9CA3AF">${Math.round(dr.value*100)}%</span></div>
      <div class="bar ${dr.severity}"><i style="width:${Math.round(dr.value*100)}%"></i></div>
    `;
    driversEl.appendChild(wrap);
  });

  // Progress bars
  const progEl = d.getElementById('progressList');
  data.progress.forEach(p => {
    const el = d.createElement('div');
    el.className = 'progress';
    el.innerHTML = `
      <div class="label">${p.label}</div>
      <div class="scale">
        <i class="red" style="width:${p.red}%"></i>
        <i class="blue" style="width:${p.blue}%; position:relative; top:-8px"></i>
      </div>
    `;
    progEl.appendChild(el);
  });

  // Tabla alumnos
  const tbody = d.querySelector('#studentsTable tbody');

  const badgeForRisk = (risk) => {
    if (risk === 'high') return 'badge red';
    if (risk === 'medium') return 'badge yellow';
    return 'badge green';
  };

  data.students.forEach(st => {
    const tr = d.createElement('tr');
    tr.className = 'row';
    tr.innerHTML = `
      <td>
        <div style="display:flex; align-items:center; gap:10px">
          <span class="avatar" style="width:28px;height:28px;border-radius:50%;display:grid;place-items:center;background:#12203f;border:1px solid rgba(255,255,255,.08);font-size:12px">
            ${st.name.split(' ').map(n => n[0]).join('').slice(0,2).toUpperCase()}
          </span>
          <div>
            <div style="font-weight:600">${st.name}</div>
            <div style="color:#9CA3AF; font-size:12px">${st.email}</div>
          </div>
        </div>
      </td>
      <td><span class="badge">${st.reasons}</span></td>
      <td>$${st.payment}</td>
      <td><span class="badge">${st.lastPayment}</span></td>
      <td><span class="${badgeForRisk(st.risk)}">${st.status}</span></td>
      <td>${st.nextEvent}</td>
      <td><button class="btn view-btn">Ver</button></td>
    `;
    tbody.appendChild(tr);
  });

  // Drawer (Plan of Action)
  const drawer = d.getElementById('drawer');
  const drawerClose = d.getElementById('drawerClose');
  const planList = d.getElementById('planList');
  const suggestBtn = d.getElementById('suggestBtn');

  const openDrawer = (plan = data.planTemplate) => {
    planList.innerHTML = '';
    plan.forEach(step => {
      const li = d.createElement('li');
      li.textContent = step;
      planList.appendChild(li);
    });
    drawer.classList.add('open');
  };
  const closeDrawer = () => drawer.classList.remove('open');

  drawerClose.addEventListener('click', closeDrawer);
  suggestBtn.addEventListener('click', () => {
    // Aquí podrías aplicar lógica de IA/score para personalizar los pasos
    openDrawer([
      'Enviar WhatsApp empático (última asistencia baja)',
      'Ofrecer plan flexible por 2 semanas',
      'Agendar tutoría 1:1',
      'Recordatorio automático si no responde en 48h',
      'Escalar a llamada del coordinador'
    ]);
  });

  // Acciones "Ver" → abre el drawer con plan
  d.querySelectorAll('.view-btn').forEach(btn => {
    btn.addEventListener('click', () => openDrawer());
  });
})();
