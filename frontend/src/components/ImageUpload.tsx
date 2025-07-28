'use client';

import React, { useRef, useState, useImperativeHandle, forwardRef } from 'react';

export type ImageUploadRef = {
  getImageData: () => string | null;
  clearImage: () => void;
};

interface ImageUploadProps {
  acceptedTypes?: string;
  maxSizeMB?: number;
}

const ImageUpload = forwardRef<ImageUploadRef, ImageUploadProps>(({ 
  acceptedTypes = "image/*", 
  maxSizeMB = 5 
}, ref) => {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [fileName, setFileName] = useState<string>('');
  const [error, setError] = useState<string>('');

  const clearImage = () => {
    setSelectedImage(null);
    setFileName('');
    setError('');
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  // Expose functions to parent
  useImperativeHandle(ref, () => ({
    getImageData: () => selectedImage,
    clearImage
  }), [selectedImage]);

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    setError('');

    // Check file size
    if (file.size > maxSizeMB * 1024 * 1024) {
      setError(`El archivo es demasiado grande. Máximo ${maxSizeMB}MB permitido.`);
      return;
    }

    // Check file type
    if (!file.type.startsWith('image/')) {
      setError('Por favor selecciona un archivo de imagen válido.');
      return;
    }

    const reader = new FileReader();
    reader.onload = (e) => {
      const result = e.target?.result as string;
      setSelectedImage(result);
      setFileName(file.name);
    };
    reader.onerror = () => {
      setError('Error al leer el archivo. Por favor intenta de nuevo.');
    };
    reader.readAsDataURL(file);
  };

  const handleDrop = (event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    const file = event.dataTransfer.files[0];
    if (file) {
      // Handle file directly instead of simulating event
      handleFileDirectly(file);
    }
  };

  const handleFileDirectly = (file: File) => {
    setError('');

    // Check file size
    if (file.size > maxSizeMB * 1024 * 1024) {
      setError(`El archivo es demasiado grande. Máximo ${maxSizeMB}MB permitido.`);
      return;
    }

    // Check file type
    if (!file.type.startsWith('image/')) {
      setError('Por favor selecciona un archivo de imagen válido.');
      return;
    }

    const reader = new FileReader();
    reader.onload = (e) => {
      const result = e.target?.result as string;
      setSelectedImage(result);
      setFileName(file.name);
    };
    reader.onerror = () => {
      setError('Error al leer el archivo. Por favor intenta de nuevo.');
    };
    reader.readAsDataURL(file);
  };

  const handleDragOver = (event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
  };

  return (
    <div className="flex flex-col items-center space-y-4">
      {/* Upload Area */}
      <div
        className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors duration-200 ${
          selectedImage 
            ? 'border-green-300 bg-green-50' 
            : 'border-gray-300 bg-gray-50 hover:border-gray-400 hover:bg-gray-100'
        }`}
        onClick={() => fileInputRef.current?.click()}
        onDrop={handleDrop}
        onDragOver={handleDragOver}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept={acceptedTypes}
          onChange={handleFileSelect}
          className="hidden"
        />
        
        {selectedImage ? (
          <div className="space-y-4">
            <div className="relative">
              <img
                src={selectedImage}
                alt="Imagen seleccionada"
                className="max-w-full max-h-64 mx-auto rounded-lg shadow-md"
              />
            </div>
            <div className="text-sm text-green-700">
              <p className="font-medium">Imagen cargada:</p>
              <p className="truncate max-w-xs mx-auto">{fileName}</p>
            </div>
          </div>
        ) : (
          <div className="space-y-4">
            <div className="text-gray-400">
              <svg className="mx-auto h-12 w-12" stroke="currentColor" fill="none" viewBox="0 0 48 48">
                <path d="M28 8H12a4 4 0 00-4 4v20m32-12v8m0 0v8a4 4 0 01-4 4H12a4 4 0 01-4-4v-4m32-4l-3.172-3.172a4 4 0 00-5.656 0L28 28M8 32l9.172-9.172a4 4 0 015.656 0L28 28m0 0l4 4m4-24h8m-4-4v8m-12 4h.02" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round" />
              </svg>
            </div>
            <div className="text-gray-600">
              <p className="text-lg font-medium">Selecciona una imagen</p>
              <p className="text-sm">o arrastra y suelta aquí</p>
              <p className="text-xs text-gray-500 mt-2">
                PNG, JPG, JPEG hasta {maxSizeMB}MB
              </p>
            </div>
          </div>
        )}
      </div>

      {/* Error Message */}
      {error && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-3 text-center">
          <p className="text-red-800 text-sm font-medium">{error}</p>
        </div>
      )}

      {/* Clear Button */}
      {selectedImage && (
        <button 
          className="px-6 py-2 bg-gray-600 text-white rounded-lg hover:bg-gray-700 transition-colors duration-200 font-medium"
          onClick={clearImage}
        >
          Limpiar Imagen
        </button>
      )}
    </div>
  );
});

ImageUpload.displayName = 'ImageUpload';

export default ImageUpload;
