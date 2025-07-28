'use client';

import React, { useRef, useState } from 'react';
import ImageUpload, { ImageUploadRef } from '../../components/ImageUpload';
import Link from 'next/link';

const BloodMNISTPage = () => {
  const imageUploadRef = useRef<ImageUploadRef>(null);
  const [prediction, setPrediction] = useState<string | null>(null);
  const [confidence, setConfidence] = useState<number | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string>('');

  const bloodCellTypes = [
    'Basophil', 'Eosinophil', 'Erythroblast', 'Immature Granulocytes',
    'Lymphocyte', 'Monocyte', 'Neutrophil', 'Platelet'
  ];

  const handlePredict = async () => {
    if (!imageUploadRef.current) return;
    
    const imageData = imageUploadRef.current.getImageData();
    
    if (!imageData) {
      setError('Por favor, selecciona una imagen de células sanguíneas para analizar.');
      return;
    }

    setError('');
    setPrediction(null);
    setConfidence(null);
    setLoading(true);

    try {
      const response = await fetch('/api/predict/bloodmnist', {
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
      setPrediction(data.prediction);
      setConfidence(data.confidence);
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
    <main className="min-h-screen bg-gradient-to-br from-red-50 to-red-100 py-8 px-4">
      <div className="max-w-2xl mx-auto">
        {/* Header */}
        <div className="text-center mb-8">
          <Link 
            href="/" 
            className="inline-flex items-center text-red-600 hover:text-red-800 mb-4 transition-colors"
          >
            ← Volver al inicio
          </Link>
          <h1 className="text-4xl font-bold text-gray-900 mb-2">
            BloodMNIST - Clasificación de Células Sanguíneas
          </h1>
          <p className="text-lg text-gray-600">
            Sube una imagen de células sanguíneas para clasificar el tipo de célula
          </p>
        </div>

        {/* Main Content */}
        <div className="bg-white rounded-2xl shadow-lg p-8">
          <div className="flex flex-col items-center space-y-6">
            
            {/* Image Upload */}
            <div className="text-center w-full">
              <h2 className="text-xl font-semibold text-gray-800 mb-4">
                Selecciona una imagen de células sanguíneas
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
                className="px-8 py-3 bg-red-600 text-white rounded-lg hover:bg-red-700 disabled:bg-red-400 disabled:cursor-not-allowed transition-colors duration-200 font-medium"
              >
                {loading ? (
                  <div className="flex items-center space-x-2">
                    <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                    <span>Analizando...</span>
                  </div>
                ) : (
                  'Clasificar Célula'
                )}
              </button>
            </div>

            {/* Results */}
            {prediction && (
              <div className="w-full max-w-md">
                <div className="bg-red-50 border border-red-200 rounded-lg p-6 text-center">
                  <h3 className="text-lg font-semibold text-red-800 mb-2">
                    Resultado de la Clasificación
                  </h3>
                  <div className="text-2xl font-bold text-red-700 mb-2">
                    {prediction}
                  </div>
                  {confidence !== null && (
                    <p className="text-sm text-red-600">
                      Confianza: {(confidence * 100).toFixed(1)}%
                    </p>
                  )}
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
              <div className="bg-red-50 border border-red-200 rounded-lg p-4">
                <h4 className="font-semibold text-red-800 mb-2">
                  Tipos de células que puede clasificar:
                </h4>
                <div className="grid grid-cols-2 gap-1 text-xs text-red-700">
                  {bloodCellTypes.map((cellType, index) => (
                    <div key={index} className="text-left">
                      • {cellType}
                    </div>
                  ))}
                </div>
                <div className="mt-3 text-xs text-red-600">
                  <p>• Sube una imagen clara de células sanguíneas</p>
                  <p>• Formatos soportados: PNG, JPG, JPEG</p>
                  <p>• Tamaño máximo: 10MB</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </main>
  );
};

export default BloodMNISTPage;
