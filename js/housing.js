import * as tf from '@tensorflow/tfjs';

let model = null;
let minInput = null;
let maxInput = null;

// Дані для тренування (спрощена версія)
const TRAINING_DATA = {
  inputs: [
    [3225, 3], [3789, 3], [3636, 3], [3109, 3], [3836, 3],
    [2998, 2], [2732, 2], [3899, 2], [2594, 2], [3068, 2],
    [1662, 1], [1456, 1], [1966, 1], [2158, 1], [1717, 1],
    [1529, 0], [2600, 0], [1271, 0], [1990, 0], [1792, 0]
  ],
  outputs: [
    262300, 270000, 262000, 259700, 259000,
    195000, 185000, 174000, 179000, 172900,
    130000, 125000, 135000, 140000, 132000,
    100000, 120000, 95000, 110000, 105000
  ]
};

export async function initHousing() {
  const statusEl = document.getElementById('housing-status');
  
  try {
    // Створення тензорів
    const inputTensor = tf.tensor2d(TRAINING_DATA.inputs);
    const outputTensor = tf.tensor1d(TRAINING_DATA.outputs);
    
    // Нормалізація
    minInput = tf.min(inputTensor, 0);
    maxInput = tf.max(inputTensor, 0);
    
    const normalize = (tensor) => {
      return tf.tidy(() => {
        const range = tf.sub(maxInput, minInput);
        return tf.div(tf.sub(tensor, minInput), range);
      });
    };
    
    const normalizedInput = normalize(inputTensor);
    
    // Створення моделі
    model = tf.sequential();
    model.add(tf.layers.dense({
      inputShape: [2],
      units: 16,
      activation: 'relu'
    }));
    model.add(tf.layers.dense({
      units: 8,
      activation: 'relu'
    }));
    model.add(tf.layers.dense({ units: 1 }));
    
    // Компіляція
    model.compile({
      optimizer: tf.train.adam(0.01),
      loss: 'meanSquaredError'
    });
    
    // Навчання
    statusEl.textContent = 'Навчання моделі...';
    await model.fit(normalizedInput, outputTensor, {
      epochs: 100,
      batchSize: 4,
      shuffle: true,
      callbacks: {
        onEpochEnd: (epoch, logs) => {
          if (epoch % 20 === 0) {
            console.log(`Epoch ${epoch}: loss = ${logs.loss.toFixed(4)}`);
          }
        }
      }
    });
    
    statusEl.textContent = '✅ Модель готова';
    statusEl.style.color = '#00ff88';
    
    // Очищення
    inputTensor.dispose();
    outputTensor.dispose();
    normalizedInput.dispose();
    
    // Обробка кнопки прогнозу
    document.getElementById('predict-btn').addEventListener('click', predictPrice);
    
    // Малювання графіка
    drawChart();
    
  } catch (error) {
    console.error('Помилка ініціалізації:', error);
    statusEl.textContent = '❌ Помилка завантаження';
    statusEl.style.color = '#ff4757';
  }
}

async function predictPrice() {
  const size = parseFloat(document.getElementById('house-size').value);
  const bedrooms = parseInt(document.getElementById('house-bedrooms').value);
  const resultEl = document.getElementById('housing-result');
  
  if (!size || size < 500 || size > 5000) {
    resultEl.textContent = 'Введіть коректну площу (500-5000)';
    resultEl.style.fontSize = '1rem';
    return;
  }
  
  const normalize = (tensor) => {
    return tf.tidy(() => {
      const range = tf.sub(maxInput, minInput);
      return tf.div(tf.sub(tensor, minInput), range);
    });
  };
  
  const input = tf.tensor2d([[size, bedrooms]]);
  const normalized = normalize(input);
  const prediction = model.predict(normalized);
  const price = prediction.dataSync()[0];
  
  resultEl.textContent = `$${Math.round(price).toLocaleString()}`;
  resultEl.style.fontSize = '2rem';
  
  input.dispose();
  normalized.dispose();
  prediction.dispose();
}

function drawChart() {
  const canvas = document.getElementById('housing-chart');
  const ctx = canvas.getContext('2d');
  
  // Адаптація розміру
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;
  ctx.scale(dpr, dpr);
  
  const width = rect.width;
  const height = rect.height;
  const padding = 50;
  
  // Очищення
  ctx.clearRect(0, 0, width, height);
  
  // Сітка
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.1)';
  ctx.lineWidth = 1;
  
  for (let i = 0; i <= 5; i++) {
    const y = padding + (i * (height - 2 * padding) / 5);
    ctx.beginPath();
    ctx.moveTo(padding, y);
    ctx.lineTo(width - padding, y);
    ctx.stroke();
  }
  
  // Точки даних
  ctx.fillStyle = '#00d9ff';
  TRAINING_DATA.inputs.forEach((input, i) => {
    const x = padding + (input[0] / 5000) * (width - 2 * padding);
    const y = height - padding - (TRAINING_DATA.outputs[i] / 300000) * (height - 2 * padding);
    
    ctx.beginPath();
    ctx.arc(x, y, 5, 0, Math.PI * 2);
    ctx.fill();
  });
  
  // Підписи
  ctx.fillStyle = '#888';
  ctx.font = '12px sans-serif';
  ctx.textAlign = 'center';
  ctx.fillText('Площа (кв. фути)', width / 2, height - 10);
  
  ctx.save();
  ctx.translate(15, height / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText('Ціна ($)', 0, 0);
  ctx.restore();
}