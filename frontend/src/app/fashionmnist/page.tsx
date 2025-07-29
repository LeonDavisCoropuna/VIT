'use client';

import React, { useRef, useState } from 'react';
import ImageUpload, { ImageUploadRef } from '../../components/ImageUpload';
import Link from 'next/link';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog';

const FashionMNISTPage = () => {
  const imageUploadRef = useRef<ImageUploadRef>(null);
  const [prediction, setPrediction] = useState<string | null>(null);
  const [confidence, setConfidence] = useState<number | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string>('');
  const [selectedImage, setSelectedImage] = useState<{ src: string; alt: string } | null>(null);

  const fashionItems = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
  ];

  const handleImageClick = (src: string, alt: string) => {
    setSelectedImage({ src, alt });
  };

  const handlePredict = async () => {
    if (!imageUploadRef.current) return;

    const imageData = imageUploadRef.current.getImageData();

    if (!imageData) {
      setError('Por favor, selecciona una imagen de ropa o accesorio para clasificar.');
      return;
    }

    setError('');
    setPrediction(null);
    setConfidence(null);
    setLoading(true);

    try {
      const response = await fetch('/api/predict/fashionmnist', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ imageData })
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Error en la predicción del modelo.');
      }


      const data = await response.json();
      setPrediction(data.digits?.[0] ?? null);
    } catch (err: any) {
      setError(err.message || 'Ocurrió un error inesperado.');
    } finally {
      setLoading(false);
    }
  };

  const handleClear = () => {
    if (imageUploadRef.current) {
      imageUploadRef.current.clearImage();
    }
    setPrediction(null);
    setConfidence(null);
    setError('');
  };

  return (
    <main className="min-h-screen bg-gradient-to-br from-purple-50 to-purple-100 py-8 px-4">
      <div className="max-w-7xl mx-auto space-y-8">
        {/* Header */}
        <div className="text-center mb-8">
          <Link
            href="/"
            className="inline-flex items-center text-purple-600 hover:text-purple-800 mb-4 transition-colors"
          >
            ← Volver al inicio
          </Link>
          <h1 className="text-4xl font-bold text-gray-900 mb-2">
            FashionMNIST - Clasificación de Ropa
          </h1>
          <p className="text-lg text-gray-600">
            Sube una imagen de ropa o accesorio para identificar el tipo de prenda
          </p>
        </div>

        {/* Main Content */}
        <div className="bg-white rounded-2xl shadow-lg p-8 flex flex-col md:flex-row gap-12">
          <div className="flex flex-col items-center space-y-6">

            {/* Image Upload */}
            <div className="text-center w-full">
              <h2 className="text-2xl font-bold text-gray-900 mb-6 text-center">
                Selecciona una imagen de ropa o accesorio
              </h2>
              <ImageUpload ref={imageUploadRef} maxSizeMB={10} />
            </div>

            {/* Action Buttons */}
            <div className="flex space-x-4">
              <button
                onClick={handleClear}
                className="px-6 py-3 bg-gray-500 text-white rounded-lg hover:bg-gray-600 transition-colors duration-200 font-medium"
              >
                Limpiar Todo
              </button>
              <button
                onClick={handlePredict}
                disabled={loading}
                className="px-8 py-3 bg-purple-600 text-white rounded-lg hover:bg-purple-700 disabled:bg-purple-400 disabled:cursor-not-allowed transition-colors duration-200 font-medium"
              >
                {loading ? (
                  <div className="flex items-center space-x-2">
                    <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                    <span>Clasificando...</span>
                  </div>
                ) : (
                  'Clasificar Prenda'
                )}
              </button>
            </div>

            {/* Results */}
            {prediction !== null && (
              <div className="w-full max-w-md">
                <div className="bg-green-50 border border-green-200 rounded-lg p-6 text-center">
                  <h3 className="text-lg font-semibold text-green-800 mb-2">
                    Resultado de la Predicción
                  </h3>
                  <div className="text-4xl font-bold text-green-700 mb-2">
                    {prediction}
                  </div>
                </div>
              </div>
            )}

            {/* Error Message */}
            {error && (
              <div className="w-full max-w-md">
                <div className="bg-red-50 border border-red-200 rounded-lg p-4 text-center">
                  <p className="text-red-800 font-medium">
                    {error}
                  </p>
                </div>
              </div>
            )}

            {/* Instructions */}
            <div className="w-full max-w-md text-center">
              <div className="bg-purple-50 border border-purple-200 rounded-lg p-4">
                <h4 className="font-semibold text-purple-800 mb-2">
                  Tipos de prendas que puede clasificar:
                </h4>
                <div className="grid grid-cols-2 gap-1 text-xs text-purple-700">
                  {fashionItems.map((item, index) => (
                    <div key={index} className="text-left">
                      • {item}
                    </div>
                  ))}
                </div>
                <div className="mt-3 text-xs text-purple-600">
                  <p>• Sube una imagen clara de la prenda</p>
                  <p>• Formatos soportados: PNG, JPG, JPEG</p>
                  <p>• Tamaño máximo: 10MB</p>
                </div>
              </div>
            </div>

          </div>
          {/* Model Performance Section */}
          <div className="flex-1">
            <h2 className="text-2xl font-bold text-gray-900 mb-6 text-center">
              Desempeño del Modelo
            </h2>
            <div className="space-y-6">
              {/* Metrics */}
              <div className="bg-gray-50 border border-gray-200 rounded-lg p-6">
                <h3 className="text-lg font-semibold text-gray-800 mb-4">
                  Métricas de Evaluación
                </h3>
                <table className="min-w-full text-sm text-left text-gray-700">
                  <tbody className="bg-white border border-gray-200 rounded-lg">
                    <tr className="border-b">
                      <th className="px-4 py-2 font-medium">Tipo</th>
                      <td className="px-4 py-2">TEST Evaluation</td>
                    </tr>
                    <tr className="border-b">
                      <th className="px-4 py-2 font-medium">Loss</th>
                      <td className="px-4 py-2">0.3662</td>
                    </tr>
                    <tr className="border-b">
                      <th className="px-4 py-2 font-medium">Accuracy</th>
                      <td className="px-4 py-2">87.0707%</td>
                    </tr>
                    <tr className="border-b">
                      <th className="px-4 py-2 font-medium">F1 Score (macro)</th>
                      <td className="px-4 py-2">0.8697</td>
                    </tr>
                    <tr className="border-b">
                      <th className="px-4 py-2 font-medium">F1 Score (weighted)</th>
                      <td className="px-4 py-2">0.8697</td>
                    </tr>
                    <tr className="border-b">
                      <th className="px-4 py-2 font-medium">Precision (macro)</th>
                      <td className="px-4 py-2">0.8706</td>
                    </tr>
                    <tr>
                      <th className="px-4 py-2 font-medium">Recall (macro)</th>
                      <td className="px-4 py-2">0.8707</td>
                    </tr>
                  </tbody>
                </table>
                {/* Images */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  {[
                    { src: '/fashion/confusion-matrix.png', alt: 'Matriz de Confusión', title: 'Matriz de Confusión' },
                    { src: '/fashion/metrics-class.png', alt: 'Métricas por Clase', title: 'Métricas por Clase' },
                    { src: '/fashion/loss-curve.png', alt: 'Curva de Pérdida', title: 'Curva de Pérdida' },
                    { src: '/fashion/accuracy-curve.png', alt: 'Curva de Precisión', title: 'Curva de Precisión' },
                  ].map((image, index) => (
                    <Dialog key={index}>
                      <DialogTrigger asChild>
                        <div
                          className="flex flex-col items-center cursor-pointer"
                          onClick={() => handleImageClick(image.src, image.alt)}
                        >
                          <p className="mt-2 text-md font-bold text-gray-600">{image.title}</p>
                          <img
                            src={image.src}
                            alt={image.alt}
                            className="w-full h-auto rounded-lg shadow-md transform hover:scale-105 transition-transform duration-200"
                          />
                        </div>
                      </DialogTrigger>
                      <DialogContent className="max-w-3xl min-w-4xl p-6">
                        <DialogHeader>
                          <DialogTitle className='text-center'>{image.title}</DialogTitle>
                        </DialogHeader>
                        <img
                          src={image.src}
                          alt={image.alt}
                          className="w-full h-auto rounded-lg"
                        />
                      </DialogContent>
                    </Dialog>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </main >
  );
};

export default FashionMNISTPage;
