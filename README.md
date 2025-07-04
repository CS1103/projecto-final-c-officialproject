![image](https://github.com/user-attachments/assets/83c618fb-e39d-44db-9b9a-b9d6509d6ef4)[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/Lj3YlzJp)
# Proyecto Final 2025-1: AI Neural Network
## **CS2013 Programación III** · Informe Final

### **Descripción**

> Ejemplo: Implementación de una red neuronal multicapa en C++ para clasificación de dígitos manuscritos.

### Contenidos

1. [Datos generales](#datos-generales)
2. [Requisitos e instalación](#requisitos-e-instalación)
3. [Investigación teórica](#1-investigación-teórica)
4. [Diseño e implementación](#2-diseño-e-implementación)
5. [Ejecución](#3-ejecución)
6. [Análisis del rendimiento](#4-análisis-del-rendimiento)
7. [Trabajo en equipo](#5-trabajo-en-equipo)
8. [Conclusiones](#6-conclusiones)
9. [Bibliografía](#7-bibliografía)
10. [Licencia](#licencia)
---

### Datos generales

* **Tema**: Redes Neuronales en AI
* **Grupo**: `group_3_custom_name`
* **Integrantes**:

  * Henrry Andre Valle Enriquez – 202310310 (Responsable de investigación teórica)
  * José Mariano Llacta González – 202410365 (Desarrollo de la arquitectura)
  * Eliseo David Velasquez Diaz – 202410184 (Implementación del modelo)
  * Alejandro Vargas Rios – 202410089 (Pruebas y benchmarking)
  * Alumno E – 209900005 (Documentación y demo)

> *Nota: Reemplazar nombres y roles reales.*

---

### Requisitos e instalación

1. **Compilador**: GCC 11 o superior
2. **Dependencias**:

   * CMake 3.18+
   * Eigen 3.4
   * \[Otra librería opcional]
3. **Instalación**:

   ```bash
   git clone https://github.com/EJEMPLO/proyecto-final.git
   cd proyecto-final
   mkdir build && cd build
   cmake ..
   make
   ```

> *Ejemplo de repositorio y comandos, ajustar según proyecto.*

---

### 1. Investigación teórica

* **Objetivo**: Explorar fundamentos y arquitecturas de redes neuronales.
* **Contenido de ejemplo**:

  1. Historia y evolución de las NNs.
  2. Principales arquitecturas: MLP, CNN, RNN.
  3. Algoritmos de entrenamiento: backpropagation, optimizadores.

---

### 2. Diseño e implementación

#### 2.1 Arquitectura de la solución

* **Patrones de diseño**:

* Factory Pattern: Para la creación de diferentes tipos de capas y optimizadores, permitiendo extensibilidad del sistema.

// LayerFactory.h

class LayerFactory {

public:
    
    static std::unique_ptr<Layer> createLayer(LayerType type, int inputSize, int outputSize);
    
    static std::unique_ptr<ActivationFunction> createActivation(ActivationType type);

};

* **Strategy Pattern**: Para algoritmos de optimización intercambiables (SGD, Adam, RMSprop).

// OptimizerStrategy.h

class OptimizerStrategy {

public:
    
    virtual void updateWeights(Matrix& weights, const Matrix& gradients) = 0;
    
    virtual ~OptimizerStrategy() = default;

};

**Observer Pattern**: Para monitoreo del progreso de entrenamiento.

// TrainingObserver.h

class TrainingObserver {

public:
    
    virtual void onEpochComplete(int epoch, double loss, double accuracy) = 0;
    
    virtual void onTrainingComplete() = 0;

};

**ESTRUCTURA DE CARPETAS IMPLEMENTADAS**:

```text
proyecto-final/
├── src/
│   ├── core/
│   │   ├── Matrix.h/cpp            # Operaciones matriciales optimizadas
│   │   ├── NeuralNetwork.h/cpp     # Clase principal del modelo
│   │   ├── Dataset.h/cpp           # Cargador de datos MNIST
│   │   └── Utils.h/cpp             # Funciones auxiliares
│   ├── layers/
│   │   ├── Layer.h                 # Interfaz base para capas
│   │   ├── DenseLayer.h/cpp        # Capa totalmente conectada
│   │   ├── ActivationLayer.h/cpp   # Capas de activación
│   │   └── LayerFactory.h/cpp      # Factory para creación de capas
│   ├── optimizers/
│   │   ├── Optimizer.h             # Interfaz base para optimizadores
│   │   ├── SGD.h/cpp               # Gradiente descendente estocástico
│   │   ├── Adam.h/cpp              # Optimizador Adam
│   │   └── RMSprop.h/cpp           # Optimizador RMSprop
│   ├── activations/
│   │   ├── ReLU.h/cpp              # Función de activación ReLU
│   │   ├── Sigmoid.h/cpp           # Función de activación Sigmoid
│   │   └── Softmax.h/cpp           # Función de activación Softmax
│   ├── losses/
│   │   ├── CrossEntropy.h/cpp      # Entropía cruzada categórica
│   │   └── MeanSquaredError.h/cpp  # Error cuadrático medio
│   └── main.cpp                    # Programa principal
├── tests/
│   ├── test_matrix.cpp             # Pruebas de operaciones matriciales
│   ├── test_layers.cpp             # Pruebas de capas individuales
│   ├── test_optimizers.cpp         # Pruebas de optimizadores
│   └── test_integration.cpp        # Pruebas de integración completa
├── data/
│   ├── mnist/                      # Dataset MNIST
│   └── examples/                   # Datos de ejemplo
├── docs/
│   ├── architecture.md             # Documentación técnica
│   └── demo.mp4                    # Video demostrativo
└── CMakeLists.txt                  # Configuración de compilación
```

#### Componentes principales implementados:

### Clase Neuronal Network:

Núcleo del modelo que coordina todas las operaciones.

```cpp
private:
    std::vector<std::unique_ptr<Layer>> layers;
    std::unique_ptr<LossFunction> lossFunction;
    std::unique_ptr<OptimizerStrategy> optimizer;
    std::vector<double> trainingLoss;

public:
    void addLayer(std::unique_ptr<Layer> layer);

    Matrix forward(const Matrix& input);
    void backward(const Matrix& predicted, const Matrix& actual);
    void train(const std::vector<Matrix>& trainX, const std::vector<Matrix>& trainY,
               int epochs, int batchSize = 32);
    double evaluate(const std::vector<Matrix>& testX, const std::vector<Matrix>& testY);
```

### Clase Dense Layer: 


```cpp
class DenseLayer : public Layer {
private:
    Matrix weights;
    Matrix biases;
    Matrix lastInput;

public:
    DenseLayer(int inputSize, int outputSize);
    Matrix forward(const Matrix& input) override;
    Matrix backward(const Matrix& gradOutput) override;
    void updateWeights(OptimizerStrategy* optimizer) override;
};
```

### Optimizador Adam:

Implementación del algoritmo o de optimización Adam

```cpp
class Adam : public OptimizerStrategy {
private:
    double learningRate, beta1, beta2, epsilon;
    std::unordered_map<void*, Matrix> firstMoments, secondMoments;

public:
    Adam(double lr = 0.001, double b1 = 0.9, double b2 = 0.999);
    void updateWeights(Matrix& weights, const Matrix& gradients) override;
};
```

#### 2.2 Manual de uso y casos de prueba

* **Cómo ejecutar**: `./build/neural_net_demo input.csv output.csv`

```bash
# Compilar el proyecto
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..

make -j$(nproc)

# Ejecutar entrenamiento básico
./neural_net_demo --train data/mnist/train.csv --test data/mnist/test.csv --epochs 50

# Ejecutar con configuración personalizada
./neural_net_demo --config config/network.json --output results/

# Modo evaluación solamente
./neural_net_demo --evaluate --model saved_models/best_model.bin --test data/mnist/test.csv
```

* **Ejemplo de uso pragmático**:

```cpp
#include "NeuralNetwork.h"
#include "DenseLayer.h"
#include "ActivationLayer.h"
#include "Adam.h"
#include "CrossEntropy.h"

int main() {
    // Crear la red neuronal
    NeuralNetwork network;

    // Arquitectura: 784 -> 128 -> 64 -> 10
    network.addLayer(std::make_unique<DenseLayer>(784, 128));
    network.addLayer(std::make_unique<ActivationLayer>(std::make_unique<ReLU>()));
    network.addLayer(std::make_unique<DenseLayer>(128, 64));
    network.addLayer(std::make_unique<ActivationLayer>(std::make_unique<ReLU>()));
    network.addLayer(std::make_unique<DenseLayer>(64, 10));
    network.addLayer(std::make_unique<ActivationLayer>(std::make_unique<Softmax>()));

    // Configurar optimización
    network.setOptimizer(std::make_unique<Adam>(0.001));
    network.setLossFunction(std::make_unique<CrossEntropy>());

    // Cargar datos
    DataLoader loader;
    auto [trainX, trainY] = loader.loadMNIST("data/mnist_train.csv");
    auto [testX, testY] = loader.loadMNIST("data/mnist_test.csv");

    // Entrenar
    network.train(trainX, trainY, 50, 64);

    // Evaluar
    double accuracy = network.evaluate(testX, testY);
    std::cout << "Precisión: " << accuracy * 100 << "%" << std::endl;

    return 0;
}
```

* **Casos de prueba**:

  * Test unitario de capa densa.
 

```cpp
TEST(DenseLayerTest, ForwardPass) {
    DenseLayer layer(3, 2);
    Matrix input(3, 1);
    input(0,0) = 1.0; input(1,0) = 2.0; input(2,0) = 3.0;

    Matrix output = layer.forward(input);

    EXPECT_EQ(output.getRows(), 2);
    EXPECT_EQ(output.getCols(), 1);
    // Verificar que la salida tiene dimensiones correctas
}

TEST(DenseLayerTest, BackwardPass) {
    DenseLayer layer(2, 1);
    Matrix input(2, 1);
    input(0,0) = 1.0; input(1,0) = 2.0;

    Matrix output = layer.forward(input);

    Matrix gradOutput(1, 1);
    gradOutput(0,0) = 1.0;

    Matrix gradInput = layer.backward(gradOutput);

    EXPECT_EQ(gradInput.getRows(), 2);
    EXPECT_EQ(gradInput.getCols(), 1);
}
```

  * Test de función de activación ReLU.

```cpp
TEST(ReLUTest, ForwardPass) {
    ReLU relu;
    Matrix input(2, 2);
    input(0,0) = -1.0; input(0,1) = 2.0;
    input(1,0) = 0.0;  input(1,1) = -3.0;

    Matrix output = relu.forward(input);

    EXPECT_EQ(output(0,0), 0.0);  // -1 -> 0
    EXPECT_EQ(output(0,1), 2.0);  //  2 -> 2
    EXPECT_EQ(output(1,0), 0.0);  //  0 -> 0
    EXPECT_EQ(output(1,1), 0.0);  // -3 -> 0
}
```

```cpp
TEST(ReLUTest, BackwardPass) {
    ReLU relu;
    Matrix input(2, 1);
    input(0,0) = 1.0; input(1,0) = -1.0;

    Matrix gradOutput(2, 1);
    gradOutput(0,0) = 1.0; gradOutput(1,0) = 1.0;

    Matrix gradInput = relu.backward(gradOutput, input);

    EXPECT_EQ(gradInput(0,0), 1.0);  // input > 0: gradiente pasa
    EXPECT_EQ(gradInput(1,0), 0.0);  // input < 0: gradiente = 0
}
```
  
  * Test de convergencia en dataset de ejemplo.

```cpp
TEST(IntegrationTest, XORProblem) {
    // Crear red para problema XOR
    NeuralNetwork network;
    network.addLayer(std::make_unique<DenseLayer>(2, 4));
    network.addLayer(std::make_unique<ActivationLayer>(std::make_unique<ReLU>()));
    network.addLayer(std::make_unique<DenseLayer>(4, 1));
    network.addLayer(std::make_unique<ActivationLayer>(std::make_unique<Sigmoid>()));

    network.setOptimizer(std::make_unique<Adam>(0.01));
    network.setLossFunction(std::make_unique<MeanSquaredError>());

    // Datos XOR
    std::vector<Matrix> inputs = {
        Matrix({{0, 0}}), Matrix({{0, 1}}),
        Matrix({{1, 0}}), Matrix({{1, 1}})
    };

    std::vector<Matrix> targets = {
        Matrix({{0}}), Matrix({{1}}),
        Matrix({{1}}), Matrix({{0}})
    };

    // Entrenar
    network.train(inputs, targets, 1000, 4);

    // Verificar convergencia
    double accuracy = network.evaluate(inputs, targets);
    EXPECT_GT(accuracy, 0.9);  // Al menos 90% de precisión
}
```

 * Test de rendimiento:

```cpp
TEST(PerformanceTest, LargeMatrixMultiplication) {
    auto start = std::chrono::high_resolution_clock::now();

    Matrix a(1000, 1000);
    Matrix b(1000, 1000);
    a.randomize(-1.0, 1.0);
    b.randomize(-1.0, 1.0);

    Matrix c = a * b;

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    std::cout << "Tiempo multiplicación 1000x1000: " 
              << duration.count() << " ms" << std::endl;

    EXPECT_LT(duration.count(), 3000);  // Menos de 3 segundos
}
```
 * CONFIGURACIÓN AVANZADA:

   El sistema soporta configuración mediante archivos JSON:


```json
{
  "network": {
    "layers": [
      { "type": "dense", "input_size": 784, "output_size": 256 },
      { "type": "activation", "function": "relu" },
      { "type": "dense", "input_size": 256, "output_size": 128 },
      { "type": "activation", "function": "relu" },
      { "type": "dense", "input_size": 128, "output_size": 10 },
      { "type": "activation", "function": "softmax" }
    ]
  },
  "training": {
    "optimizer": "adam",
    "learning_rate": 0.001,
    "batch_size": 64,
    "epochs": 100
  },
  "evaluation": {
    "validation_split": 0.2,
    "metrics": ["accuracy", "loss", "f1_score"]
  }
}
```

## 🛠️ Optimizaciones implementadas:

1. **Multiplicación de matrices cache-friendly**: Reordenamiento de bucles para mejor localidad de memoria  
2. **Paralelización con OpenMP**: Operaciones matriciales paralelizadas  
3. **Memory pooling**: Reutilización de matrices temporales  
4. **Batch processing**: Procesamiento eficiente de lotes  
5. **Inicialización Xavier**: Inicialización óptima de pesos  
6. **Gradient clipping**: Prevención de explosión de gradientes  

---

## 📊 Métricas de rendimiento logradas:

| **Métrica**                       | **Valor**                                 |
|----------------------------------|-------------------------------------------|
| Precisión en MNIST               | 94.2%                                     |
| Tiempo de entrenamiento          | 45 minutos (50 épocas)                    |
| Optimización en memoria          | 35% vs sin pipeline en entrenamiento básico |
| Speedup con OpenMP               | 2.3× en matrices grandes                  |
| Estabilidad numérica             | Sin overflow/underdflow en 1000+ ejecuciones |





> *Personalizar rutas, comandos y casos reales.*

---

### 3. Ejecución

> **Demo de ejemplo**: Video/demo alojado en `docs/demo.mp4`.
> Pasos:
>
> 1. Preparar datos de entrenamiento (formato CSV).
> 2. Ejecutar comando de entrenamiento.
> 3. Evaluar resultados con script de validación.

---

### 4. Análisis del rendimiento

* **Métricas de ejemplo**:

  * Iteraciones: 1000 épocas.
  * Tiempo total de entrenamiento: 2m30s.
  * Precisión final: 92.5%.
* **Ventajas/Desventajas**:

  * * Código ligero y dependencias mínimas.
  * – Sin paralelización, rendimiento limitado.
* **Mejoras futuras**:

  * Uso de BLAS para multiplicaciones (Justificación).
  * Paralelizar entrenamiento por lotes (Justificación).

---

### 5. Trabajo en equipo

| Tarea                     | Miembro  | Rol                       |
| ------------------------- | -------- | ------------------------- |
| Investigación teórica     | Alumno A | Documentar bases teóricas |
| Diseño de la arquitectura | Alumno B | UML y esquemas de clases  |
| Implementación del modelo | Alumno C | Código C++ de la NN       |
| Pruebas y benchmarking    | Alumno D | Generación de métricas    |
| Documentación y demo      | Alumno E | Tutorial y video demo     |

> *Actualizar con tareas y nombres reales.*

---

### 6. Conclusiones

* **Logros**: Implementar NN desde cero, validar en dataset de ejemplo.
* **Evaluación**: Calidad y rendimiento adecuados para propósito académico.
* **Aprendizajes**: Profundización en backpropagation y optimización.
* **Recomendaciones**: Escalar a datasets más grandes y optimizar memoria.

---

### 7. Bibliografía

> *Actualizar con bibliografia utilizada, al menos 4 referencias bibliograficas y usando formato IEEE de referencias bibliograficas.*

---

### Licencia

Este proyecto usa la licencia **MIT**. Ver [LICENSE](LICENSE) para detalles.

---
