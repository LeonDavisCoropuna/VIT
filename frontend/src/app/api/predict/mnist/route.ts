import { NextResponse } from 'next/server';
import { writeFile, unlink, cp, mkdir } from 'fs/promises';
import { spawn } from 'child_process';
import path from 'path';
import { randomUUID } from 'crypto';
import sharp from 'sharp';


export async function POST(request: Request) {
  let tempImagePath: string | null = null;
  let savedDebugPath: string | null = null;
  
  try {
    const { imageData } = await request.json();
    
    if (!imageData) {
      return NextResponse.json(
        { error: 'No se proporcionaron datos de imagen.' }, 
        { status: 400 }
      );
    }

    // Validate base64 image data
    if (!imageData.startsWith('data:image/png;base64,')) {
      return NextResponse.json(
        { error: 'Formato de imagen inválido. Se requiere PNG en base64.' }, 
        { status: 400 }
      );
    }

    // Extract base64 data
    const base64Data = imageData.replace(/^data:image\/png;base64,/, '');
    
    // Create temporary file for the image
    const tempId = randomUUID();
    tempImagePath = path.join('/tmp', `mnist_input_${tempId}.png`);
    const fileName = `mnist_input_${tempId}.png`;
    
    // Write image to temporary file
    await writeFile(tempImagePath, base64Data, 'base64');

        // Redimensionar a 28x28 usando sharp y sobrescribir el archivo temporal
        await sharp(tempImagePath)
        .resize(28, 28) 
        .flatten({ background: { r: 0, g: 0, b: 0 } }) // elimina transparencia, fondo negro
  .grayscale()
  .negate()
        .toFile(tempImagePath + '_resized.png');
    
        // Usar la imagen redimensionada como la nueva
        tempImagePath = tempImagePath + '_resized.png';
    
    

     // Copiar imagen a carpeta de debug permanente (public/mnist_debug)
     const debugDir = path.join(process.cwd(), '../custom_images');
   
 
     savedDebugPath = path.join(debugDir, fileName);
     await cp(tempImagePath, savedDebugPath); // copiar
 
     console.log(`Imagen guardada en: ${savedDebugPath}`);

    // TODO: Replace this section with your actual C++ MNIST model inference
    // Example of how you might call your C++ inference binary:
    /*
    const modelPath = '/path/to/your/mnist_model.bin';
    const inferenceResult = await runCppInference(modelPath, tempImagePath, 'mnist');
    */

    // For now, we'll simulate the inference with a random prediction
    const simulatedPrediction = await simulateMNISTInference();
    
    return NextResponse.json({
      prediction: simulatedPrediction.digit,
      confidence: simulatedPrediction.confidence
    });
    
  } catch (error: any) {
    console.error('Error en la API de predicción MNIST:', error);
    return NextResponse.json(
      { error: 'Error interno del servidor.' }, 
      { status: 500 }
    );
  } finally {
    // Clean up temporary file
    if (tempImagePath) {
      try {
        //await unlink(tempImagePath);
      } catch (cleanupError) {
        console.warn('No se pudo eliminar el archivo temporal:', cleanupError);
      }
    }
  }
}

// Simulated MNIST inference function - replace this with your actual model call
async function simulateMNISTInference(): Promise<{ digit: number; confidence: number }> {
  // Simulate processing time
  await new Promise(resolve => setTimeout(resolve, 800 + Math.random() * 1200));
  
  // Return random prediction for demonstration
  const digit = Math.floor(Math.random() * 10); // 0-9
  const confidence = 0.75 + Math.random() * 0.25; // Random confidence between 75-100%
  
  return { digit, confidence };
}

// Example function for calling your C++ inference binary
// Uncomment and modify this when you're ready to integrate your actual MNIST model

async function runCppInference(modelPath: string, imagePath: string, dataset: string): Promise<{ digit: number; confidence: number }> {
  return new Promise((resolve, reject) => {
    // Example command - adjust according to your C++ binary interface
    const child = spawn('./your_mnist_inference_binary', [
      '--model', modelPath,
      '--input', imagePath,
      '--dataset', dataset,
      '--output', 'json'
    ]);

    let stdout = '';
    let stderr = '';

    child.stdout.on('data', (data) => {
      stdout += data.toString();
    });

    child.stderr.on('data', (data) => {
      stderr += data.toString();
    });

    child.on('close', (code) => {
      if (code !== 0) {
        reject(new Error(`MNIST inference failed with code ${code}: ${stderr}`));
        return;
      }

      try {
        // Parse the output from your C++ binary
        // Adjust this according to your binary's output format
        const result = JSON.parse(stdout);
        resolve({
          digit: result.predicted_digit,
          confidence: result.confidence
        });
      } catch (parseError) {
        reject(new Error(`Failed to parse MNIST inference result: ${parseError}`));
      }
    });

    child.on('error', (error) => {
      reject(new Error(`Failed to start MNIST inference process: ${error.message}`));
    });
  });
}

