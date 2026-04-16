import * as tf from '@tensorflow/tfjs';

// Імпорт тренувальних даних MNIST
import {TRAINING_DATA} from 'https://storage.googleapis.com/jmstore/TensorFlowJS/EdX/TrainingData/mnist.js';

let model = null;
let isDrawing = false;
let lastX = 0;
let lastY = 0;
let trainingInProgress = false;

/**
 * Ініціалізація MNIST модуля
 * Завантажує модель, налаштовує canvas та обробники подій
 */
export async function initMNIST() {
  const statusEl = document.getElementById('mnist-status');
  const progressEl = document.getElementById('training-progress');
  
  try {
    // Створення та навчання моделі
    await trainModel(statusEl, progressEl);
    
    // Налаштування canvas
    setupCanvas();
    
    // Обробка кнопок
    document.getElementById('clear-canvas').addEventListener('click', clearCanvas);
    document.getElementById('recognize-btn').addEventListener('click', recognizeDigit);
    
  } catch (error) {
    console.error('Помилка ініціалізації MNIST:', error);
    statusEl.textContent = '❌ Помилка завантаження';
    statusEl.style.color = '#ff4757';
  }
}

/**
 * Навчання нейронної мережі на дані MNIST
 * @param {HTMLElement} statusEl - Елемент статусу
 * @param {HTMLElement} progressEl - Елемент прогресу
 */
async function trainModel(statusEl, progressEl) {
  statusEl.textContent = '⏳ Навчання моделі...';
  statusEl.style.color = '#ffb700';
  progressEl.style.display = 'block';
  
  try {
    // Перетворення тренувальних даних у тензори
    const INPUT_TENSOR = tf.tensor2d(TRAINING_DATA.inputs);
    const OUTPUT_TENSOR = tf.oneHot(
      tf.tensor1d(TRAINING_DATA.outputs, 'int32'),
      10
    );
    
    // Створення моделі нейронної мережі
    model = tf.sequential({
      layers: [
        // Вхідний шар: 784 входи (28x28 пікселів)
        tf.layers.dense({
          inputShape: [784],
          units: 128,
          activation: 'relu'
        }),
        // Dropout для запобігання перенавчанню
        tf.layers.dropout({ rate: 0.2 }),
        // Перший прихований шар
        tf.layers.dense({
          units: 64,
          activation: 'relu'
        }),
        // Dropout
        tf.layers.dropout({ rate: 0.2 }),
        // Другий прихований шар
        tf.layers.dense({
          units: 32,
          activation: 'relu'
        }),
        // Вихідний шар: 10 нейронів для цифр 0-9
        tf.layers.dense({
          units: 10,
          activation: 'softmax'
        })
      ]
    });
    
    // Компіляція моделі
    model.compile({
      optimizer: tf.train.adam(0.01),
      loss: 'categoricalCrossentropy',
      metrics: ['accuracy']
    });
    
    // Параметри навчання
    const EPOCHS = 50;
    const BATCH_SIZE = 512;
    
    // Навчання моделі
    await model.fit(INPUT_TENSOR, OUTPUT_TENSOR, {
      batchSize: BATCH_SIZE,
      epochs: EPOCHS,
      shuffle: true,
      callbacks: {
        onEpochEnd: (epoch, logs) => {
          // Оновлення прогрес-бара
          progressEl.value = (epoch / (EPOCHS - 1)) * 100;
          
          // Виведення статусу у консоль
          console.log(`Епоха ${epoch + 1}/${EPOCHS}:`, {
            loss: logs.loss.toFixed(4),
            accuracy: (logs.acc * 100).toFixed(2) + '%'
          });
          
          // Оновлення статусу на сторінці
          statusEl.textContent = 
            `⏳ Епоха ${epoch + 1}/${EPOCHS} - Точність: ${(logs.acc * 100).toFixed(1)}%`;
        }
      }
    });
    
    // Звільнення пам'яті
    INPUT_TENSOR.dispose();
    OUTPUT_TENSOR.dispose();
    
    // Успішне завершення навчання
    statusEl.textContent = '✅ Модель успішно натренована!';
    statusEl.style.color = '#00ff88';
    progressEl.style.display = 'none';
    
  } catch (error) {
    console.error('Помилка при навчанні моделі:', error);
    statusEl.textContent = '❌ Помилка навчання моделі';
    statusEl.style.color = '#ff4757';
    throw error;
  }
}

/**
 * Налаштування canvas для малювання цифр
 */
function setupCanvas() {
  const canvas = document.getElementById('digit-canvas');
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  
  // Чорний фон
  ctx.fillStyle = '#000';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  
  // Параметри малювання
  ctx.strokeStyle = '#fff';
  ctx.lineWidth = 15;
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  
  // Обробники подій миші
  canvas.addEventListener('mousedown', startDrawing);
  canvas.addEventListener('mousemove', draw);
  canvas.addEventListener('mouseup', stopDrawing);
  canvas.addEventListener('mouseout', stopDrawing);
  
  // Підтримка сенсорних екранів
  canvas.addEventListener('touchstart', handleTouch);
  canvas.addEventListener('touchmove', handleTouch);
  canvas.addEventListener('touchend', stopDrawing);
}

/**
 * Початок малювання
 * @param {MouseEvent} e - Подія миші
 */
function startDrawing(e) {
  isDrawing = true;
  [lastX, lastY] = [e.offsetX, e.offsetY];
}

/**
 * Малювання на canvas
 * @param {MouseEvent} e - Подія миші
 */
function draw(e) {
  if (!isDrawing) return;
  
  const canvas = document.getElementById('digit-canvas');
  const ctx = canvas.getContext('2d');
  
  ctx.beginPath();
  ctx.moveTo(lastX, lastY);
  ctx.lineTo(e.offsetX, e.offsetY);
  ctx.stroke();
  
  [lastX, lastY] = [e.offsetX, e.offsetY];
}

/**
 * Припинення малювання
 */
function stopDrawing() {
  isDrawing = false;
}

/**
 * Обробка дотику на сенсорних екранах
 * @param {TouchEvent} e - Подія дотику
 */
function handleTouch(e) {
  e.preventDefault();
  const touch = e.touches[0];
  const canvas = document.getElementById('digit-canvas');
  const rect = canvas.getBoundingClientRect();
  
  if (e.type === 'touchstart') {
    startDrawing({
      offsetX: touch.clientX - rect.left,
      offsetY: touch.clientY - rect.top
    });
  } else if (e.type === 'touchmove') {
    draw({
      offsetX: touch.clientX - rect.left,
      offsetY: touch.clientY - rect.top
    });
  }
}

/**
 * Очищення canvas
 */
function clearCanvas() {
  const canvas = document.getElementById('digit-canvas');
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = '#000';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  document.getElementById('mnist-result').textContent = '—';
  document.getElementById('confidence-bars').innerHTML = '';
}
/**
 * Розпізнавання цифри на canvas
 * Масштабує малюнок до 28x28 пікселів та передає до моделі
 */
async function recognizeDigit() {
  if (!model) {
    alert('Модель ще не готова. Будь ласка, дочекайтеся завершення навчання.');
    return;
  }
  
  const canvas = document.getElementById('digit-canvas');
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  const resultEl = document.getElementById('mnist-result');
  
  // Створення тимчасового canvas 28x28 пікселів
  const tempCanvas = document.createElement('canvas');
  tempCanvas.width = 28;
  tempCanvas.height = 28;
  const tempCtx = tempCanvas.getContext('2d');
  
  // Масштабування малюнку до 28x28
  tempCtx.fillStyle = '#000';
  tempCtx.fillRect(0, 0, 28, 28);
  tempCtx.drawImage(canvas, 0, 0, 28, 28);
  
  // Отримання даних пікселів
  const imageData = tempCtx.getImageData(0, 0, 28, 28);
  
  // Нормалізація даних (0-1 діапазон)
  const data = [];
  for (let i = 0; i < imageData.data.length; i += 4) {
    // Усереднення RGB каналів та нормалізація
    const grayscale = (imageData.data[i] + imageData.data[i + 1] + imageData.data[i + 2]) / 3;
    data.push(grayscale / 255);
  }
  
  // Перетворення у тензор
  const input = tf.tensor2d([data]);
  
  // Передбачення моделі
  const prediction = model.predict(input);
  const values = Array.from(prediction.dataSync());
  
  // Знаходження максимальної ймовірності
  const maxIdx = values.indexOf(Math.max(...values));
  const confidence = (values[maxIdx] * 100).toFixed(1);
  
  // Відображення результату
  resultEl.textContent = `${maxIdx} (${confidence}%)`;
  
  // Показ confidence bars для всіх цифр
  displayConfidenceBars(values);
  
  // Звільнення пам'яті
  input.dispose();
  prediction.dispose();
  
  console.log('Передбачення цифри:', {
    predicted: maxIdx,
    confidence: confidence + '%',
    allPredictions: values.map((v, i) => ({
      digit: i,
      probability: (v * 100).toFixed(2) + '%'
    }))
  });
}

/**
 * Відображення confidence bars для кожної цифри
 * @param {Array<number>} values - Масив ймовірностей (0-10)
 */
function displayConfidenceBars(values) {
  const container = document.getElementById('confidence-bars');
  container.innerHTML = '';
  
  values.forEach((value, idx) => {
    const bar = document.createElement('div');
    bar.className = 'confidence-bar';
    
    // Встановлення кольору залежно від ймовірності
    const percentage = value * 100;
    let color = '#666';
    if (percentage > 50) color = '#00ff88';
    else if (percentage > 20) color = '#ffb700';
    
    bar.innerHTML = `
      <span class="digit-label">${idx}</span>
      <div class="bar">
        <div class="fill" style="width: ${percentage}%; background-color: ${color};"></div>
      </div>
      <span class="value">${percentage.toFixed(1)}%</span>
    `;
    container.appendChild(bar);
  });
}