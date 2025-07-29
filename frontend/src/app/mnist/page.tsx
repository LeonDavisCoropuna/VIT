'use client';

import React, { useRef, useState } from 'react';
import DrawingCanvas, { CanvasRef } from '../../components/DrawingCanvas';
import Link from 'next/link';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog';

const MNISTPage = () => {
  const canvasRef = useRef<CanvasRef>(null);
  const [prediction, setPrediction] = useState<number | null>(null);
  const [confidence, setConfidence] = useState<number | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string>('');
  const [selectedImage, setSelectedImage] = useState<{ src: string; alt: string } | null>(null);

  const isCanvasEmpty = (imageData: string): boolean => {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    if (!ctx) return true;

    canvas.width = 280;
    canvas.height = 280;
    ctx.fillStyle = '#fff';
    ctx.fillRect(0, 0, 280, 280);

    const emptyCanvas = canvas.toDataURL('image/png');
    return imageData === emptyCanvas;
  };

  const handlePredict = async () => {
    if (!canvasRef.current) return;

    const imageData = canvasRef.current.getImageData();

    if (!imageData || isCanvasEmpty(imageData)) {
      setError('El lienzo está vacío. Por favor, dibuja un número del 0 al 9.');
      return;
    }

    setError('');
    setPrediction(null);
    setConfidence(null);
    setLoading(true);

    try {
      const response = await fetch('/api/predict/mnist', {
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
      setConfidence(1);
    } catch (err: any) {
      setError(err.message || 'Ocurrió un error inesperado.');
    } finally {
      setLoading(false);
    }
  };

  const handleClear = () => {
    if (canvasRef.current) {
      canvasRef.current.clearCanvas();
    }
    setPrediction(null);
    setConfidence(null);
    setError('');
  };

  const handleImageClick = (src: string, alt: string) => {
    setSelectedImage({ src, alt });
  };

  return (
    <main className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100 py-8 px-4">
      <div className="max-w-7xl mx-auto space-y-8">
        {/* Header */}
        <div className="text-center">
          <Link
            href="/"
            className="inline-flex items-center text-blue-600 hover:text-blue-800 mb-4 transition-colors"
          >
            ← Volver al inicio
          </Link>
          <h1 className="text-4xl font-bold text-gray-900 mb-2">
            MNIST - Reconocimiento de Dígitos
          </h1>
          <p className="text-lg text-gray-600">
            Dibuja un número del 0 al 9 y deja que el modelo lo reconozca
          </p>
        </div>

        {/* Interactive and Performance Sections */}
        <div className="bg-white rounded-2xl shadow-lg p-8 flex flex-col md:flex-row gap-12">
          {/* Interactive Section */}
          <div className="flex flex-col items-center space-y-6 flex-1">
            <h2 className="text-2xl font-bold text-gray-900 mb-6 text-center">
              Dibuja aquí tu número
            </h2>
            <DrawingCanvas ref={canvasRef} />

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
                className="px-8 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-blue-400 disabled:cursor-not-allowed transition-colors duration-200 font-medium"
              >
                {loading ? (
                  <div className="flex items-center space-x-2">
                    <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                    <span>Procesando...</span>
                  </div>
                ) : (
                  'Predecir Número'
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
                  <p className="text-red-800 font-medium">{error}</p>
                </div>
              </div>
            )}

            {/* Instructions */}
            <div className="w-full max-w-md text-center">
              <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
                <h4 className="font-semibold text-blue-800 mb-2">Instrucciones:</h4>
                <ul className="text-sm text-blue-700 space-y-1">
                  <li>• Dibuja un número del 0 al 9 en el lienzo</li>
                  <li>• Usa trazos claros y centrados</li>
                  <li>• Haz clic en "Predecir Número" para obtener el resultado</li>
                  <li>• Usa "Limpiar" para borrar y empezar de nuevo</li>
                </ul>
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
                      <td className="px-4 py-2">0.0977875</td>
                    </tr>
                    <tr className="border-b">
                      <th className="px-4 py-2 font-medium">Accuracy</th>
                      <td className="px-4 py-2">97.0148%</td>
                    </tr>
                    <tr className="border-b">
                      <th className="px-4 py-2 font-medium">F1 Score (macro)</th>
                      <td className="px-4 py-2">0.969945</td>
                    </tr>
                    <tr className="border-b">
                      <th className="px-4 py-2 font-medium">F1 Score (weighted)</th>
                      <td className="px-4 py-2">0.970077</td>
                    </tr>
                    <tr className="border-b">
                      <th className="px-4 py-2 font-medium">Precision (macro)</th>
                      <td className="px-4 py-2">0.970105</td>
                    </tr>
                    <tr>
                      <th className="px-4 py-2 font-medium">Recall (macro)</th>
                      <td className="px-4 py-2">0.96989</td>
                    </tr>
                  </tbody>
                </table>
              </div>

              {/* Images */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {[
                  { src: '/mnist/confusion-matrix.png', alt: 'Matriz de Confusión', title: 'Matriz de Confusión' },
                  { src: '/mnist/metrics-class.png', alt: 'Métricas por Clase', title: 'Métricas por Clase' },
                  { src: '/mnist/loss-curve.png', alt: 'Curva de Pérdida', title: 'Curva de Pérdida' },
                  { src: '/mnist/accuracy-curve.png', alt: 'Curva de Precisión', title: 'Curva de Precisión' },
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
    </main>
  );
};

export default MNISTPage;