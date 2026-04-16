import { initHousing } from './housing.js';
import { initMNIST } from './mnist.js';

// Навігація між вкладками
document.querySelectorAll('.nav-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
    
    btn.classList.add('active');
    document.getElementById(btn.dataset.tab).classList.add('active');
  });
});

// Ініціалізація обох моделей
async function init() {
  console.log('🚀 Ініціалізація TensorFlow.js Demo...');
  
  await Promise.all([
    initHousing(),
    initMNIST()
  ]);
  
  console.log('✅ Всі моделі готові!');
}

init();