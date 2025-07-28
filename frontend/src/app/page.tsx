'use client';

import React from 'react';
import Link from 'next/link';

const HomePage = () => {
  const datasets = [
    {
      id: 'mnist',
      name: 'MNIST',
      description: 'Reconocimiento de dígitos escritos a mano (0-9)',
      color: 'blue',
      bgColor: 'from-blue-50 to-blue-100',
      borderColor: 'border-blue-200',
      textColor: 'text-blue-800',
      buttonColor: 'bg-blue-600 hover:bg-blue-700',
      icon: '🔢',
      inputType: 'drawing',
      route: '/mnist'
    },
    {
      id: 'bloodmnist',
      name: 'BloodMNIST',
      description: 'Clasificación de células sanguíneas (8 tipos diferentes)',
      color: 'red',
      bgColor: 'from-red-50 to-red-100',
      borderColor: 'border-red-200',
      textColor: 'text-red-800',
      buttonColor: 'bg-red-600 hover:bg-red-700',
      icon: '🩸',
      inputType: 'upload',
      route: '/bloodmnist'
    },
    {
      id: 'fashionmnist',
      name: 'FashionMNIST',
      description: 'Clasificación de prendas de vestir y accesorios',
      color: 'purple',
      bgColor: 'from-purple-50 to-purple-100',
      borderColor: 'border-purple-200',
      textColor: 'text-purple-800',
      buttonColor: 'bg-purple-600 hover:bg-purple-700',
      icon: '👕',
      inputType: 'upload',
      route: '/fashionmnist'
    }
  ];

  return (
    <main className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100 py-8 px-4">
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <div className="text-center mb-12">
          <h1 className="text-5xl font-bold text-gray-900 mb-4">
            Prueba tus Modelos ViT Transformer
          </h1>
          <p className="text-xl text-gray-600 max-w-3xl mx-auto">
            Selecciona el dataset que quieres probar y experimenta con tu modelo de Vision Transformer entrenado
          </p>
        </div>

        {/* Dataset Cards */}
        <div className="grid md:grid-cols-3 gap-8 mb-12">
          {datasets.map((dataset) => (
            <div
              key={dataset.id}
              className={`bg-gradient-to-br ${dataset.bgColor} rounded-2xl shadow-lg overflow-hidden transform hover:scale-105 transition-all duration-300`}
            >
              <div className="p-8">
                <div className="text-center mb-6">
                  <div className="text-6xl mb-4">{dataset.icon}</div>
                  <h2 className="text-2xl font-bold text-gray-900 mb-2">
                    {dataset.name}
                  </h2>
                  <p className="text-gray-700 text-sm leading-relaxed">
                    {dataset.description}
                  </p>
                </div>

                <div className={`bg-white rounded-lg p-4 mb-6 border ${dataset.borderColor}`}>
                  <div className="flex items-center justify-between">
                    <span className="text-sm font-medium text-gray-600">
                      Tipo de entrada:
                    </span>
                    <span className={`text-sm font-semibold ${dataset.textColor}`}>
                      {dataset.inputType === 'drawing' ? 'Dibujo manual' : 'Subir imagen'}
                    </span>
                  </div>
                </div>

                <Link href={dataset.route}>
                  <button className={`w-full py-3 px-6 ${dataset.buttonColor} text-white rounded-lg font-semibold transition-colors duration-200 shadow-md hover:shadow-lg`}>
                    Probar Modelo
                  </button>
                </Link>
              </div>
            </div>
          ))}
        </div>

        {/* Features Section */}
        <div className="bg-white rounded-2xl shadow-lg p-8 mb-8">
          <h3 className="text-2xl font-bold text-gray-900 text-center mb-8">
            Características de la Aplicación
          </h3>
          <div className="grid md:grid-cols-3 gap-6">
            <div className="text-center">
              <div className="bg-blue-100 rounded-full w-16 h-16 flex items-center justify-center mx-auto mb-4">
                <span className="text-2xl">🎨</span>
              </div>
              <h4 className="font-semibold text-gray-900 mb-2">Interfaz Intuitiva</h4>
              <p className="text-gray-600 text-sm">
                Diseño moderno y fácil de usar para probar tus modelos de manera eficiente
              </p>
            </div>
            <div className="text-center">
              <div className="bg-green-100 rounded-full w-16 h-16 flex items-center justify-center mx-auto mb-4">
                <span className="text-2xl">⚡</span>
              </div>
              <h4 className="font-semibold text-gray-900 mb-2">Predicciones Rápidas</h4>
              <p className="text-gray-600 text-sm">
                Obtén resultados instantáneos con tu modelo ViT entrenado en formato .bin
              </p>
            </div>
            <div className="text-center">
              <div className="bg-purple-100 rounded-full w-16 h-16 flex items-center justify-center mx-auto mb-4">
                <span className="text-2xl">📊</span>
              </div>
              <h4 className="font-semibold text-gray-900 mb-2">Múltiples Datasets</h4>
              <p className="text-gray-600 text-sm">
                Prueba diferentes tipos de datos: dígitos, células sanguíneas y ropa
              </p>
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="text-center">
          <p className="text-gray-500">
            Aplicación para probar modelos ViT Transformer entrenados en C++
          </p>
          <p className="text-gray-400 text-sm mt-2">
            Soporta modelos en formato .bin para MNIST, BloodMNIST y FashionMNIST
          </p>
        </div>
      </div>
    </main>
  );
};

export default HomePage;
