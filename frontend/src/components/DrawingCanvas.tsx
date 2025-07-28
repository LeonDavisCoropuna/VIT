'use client';

import React, { useRef, useEffect, useState, useImperativeHandle, forwardRef } from 'react';

export type CanvasRef = {
  getImageData: () => string;
  clearCanvas: () => void;
};

const DrawingCanvas = forwardRef<CanvasRef>((_, ref) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [isDrawing, setIsDrawing] = useState(false);
  const [context, setContext] = useState<CanvasRenderingContext2D | null>(null);

  useEffect(() => {
    if (canvasRef.current) {
      const ctx = canvasRef.current.getContext('2d');
      if (ctx) {
        ctx.lineWidth = 32;
        ctx.lineCap = 'round';
        ctx.strokeStyle = '#000';
        setContext(ctx);
        // Set canvas background white
        ctx.fillStyle = '#fff';
        ctx.fillRect(0, 0, canvasRef.current.width, canvasRef.current.height);
      }
    }
  }, []);

  const clearCanvas = () => {
    if (canvasRef.current && context) {
      context.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
      // Fill with white background
      context.fillStyle = '#fff';
      context.fillRect(0, 0, canvasRef.current.width, canvasRef.current.height);
    }
  };

  // Expose functions to parent
  useImperativeHandle(ref, () => ({
    getImageData: () => {
      return canvasRef.current ? canvasRef.current.toDataURL('image/png') : '';
    },
    clearCanvas
  }), []);

  const getEventPos = (e: React.MouseEvent | React.TouchEvent) => {
    if (!canvasRef.current) return { x: 0, y: 0 };
    
    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    
    if ('touches' in e) {
      return {
        x: (e.touches[0].clientX - rect.left) * (canvas.width / rect.width),
        y: (e.touches[0].clientY - rect.top) * (canvas.height / rect.height)
      };
    }
    
    return {
      x: (e.clientX - rect.left) * (canvas.width / rect.width),
      y: (e.clientY - rect.top) * (canvas.height / rect.height)
    };
  };

  const startDrawing = (e: React.MouseEvent | React.TouchEvent) => {
    e.preventDefault();
    setIsDrawing(true);
    const { x, y } = getEventPos(e);
    if (context) {
      context.beginPath();
      context.moveTo(x, y);
    }
  };

  const draw = (e: React.MouseEvent | React.TouchEvent) => {
    e.preventDefault();
    if (!isDrawing || !context) return;
    
    const { x, y } = getEventPos(e);
    context.lineTo(x, y);
    context.stroke();
  };

  const finishDrawing = (e: React.MouseEvent | React.TouchEvent) => {
    e.preventDefault();
    setIsDrawing(false);
    if (context) {
      context.beginPath();
    }
  };

  return (
    <div className="flex flex-col items-center space-y-4">
      <div className="border-2 border-gray-300 rounded-lg p-2 bg-white">
        <canvas
          ref={canvasRef}
          className="cursor-crosshair touch-none"
          width={280}
          height={280}
          onMouseDown={startDrawing}
          onMouseMove={draw}
          onMouseUp={finishDrawing}
          onMouseLeave={finishDrawing}
          onTouchStart={startDrawing}
          onTouchMove={draw}
          onTouchEnd={finishDrawing}
        />
      </div>
      <button 
        className="px-6 py-2 bg-gray-600 text-white rounded-lg hover:bg-gray-700 transition-colors duration-200 font-medium"
        onClick={clearCanvas}
      >
        Limpiar
      </button>
    </div>
  );
});

DrawingCanvas.displayName = 'DrawingCanvas';

export default DrawingCanvas;
