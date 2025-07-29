import { NextResponse } from 'next/server';
import { writeFile, readdir, unlink, mkdir } from 'fs/promises';
import { spawn } from 'child_process';
import path from 'path';
import { randomUUID } from 'crypto';
import sharp from 'sharp';
import fs from 'fs';

export async function POST(request: Request) {
  const customImagesDir = path.join(process.cwd(), '../custom_images');
  let tempImagePath: string | null = null;

  try {
    const { imageData } = await request.json();

    if (!imageData || !imageData.startsWith('data:image/png;base64,')) {
      return NextResponse.json(
        { error: 'Formato de imagen inválido o no proporcionado.' },
        { status: 400 }
      );
    }

    // Asegurarse que la carpeta custom_images exista
    await mkdir(customImagesDir, { recursive: true });

    const tempId = randomUUID();
    const imageFileName = `mnist_input_${tempId}.png`;
    const originalImagePath = path.join(customImagesDir, `original_${imageFileName}`);
    const resizedImagePath = path.join(customImagesDir, imageFileName);

    // Guardar imagen original
    const base64Data = imageData.replace(/^data:image\/png;base64,/, '');
    await writeFile(originalImagePath, base64Data, 'base64');

    // Redimensionar a 28x28 y convertir a escala de grises
    await sharp(originalImagePath)
      .resize(28, 28)
      .flatten({ background: { r: 0, g: 0, b: 0 } })
      .grayscale()
      .negate()
      .toFile(resizedImagePath);

    tempImagePath = resizedImagePath;

    console.log(`🖼️ Imagen procesada: ${tempImagePath}`);

    // Ejecutar script C++
    const result = await runCppShellScript(tempImagePath);

    return NextResponse.json(result);

  } catch (error: any) {
    console.error('Error en la API de predicción MNIST:', error);
    return NextResponse.json({ error: 'Error interno del servidor.' }, { status: 500 });

  } finally {
    // Limpiar todos los archivos de custom_images (pero no la carpeta)
    try {
      const files = await readdir(customImagesDir);
      const unlinkPromises = files.map(file =>
        unlink(path.join(customImagesDir, file)).catch(err =>
          console.warn(`⚠️ No se pudo eliminar ${file}:`, err)
        )
      );
      await Promise.all(unlinkPromises);
      console.log('🧹 Archivos temporales eliminados de custom_images/');
    } catch (err) {
      console.warn('⚠️ No se pudo limpiar custom_images:', err);
    }
  }
}

async function runCppShellScript(imagePath: string): Promise<{ digits: number[]; count: number }> {
  return new Promise((resolve, reject) => {
    const scriptPath = path.join(process.cwd(), '../run.sh');

    console.log('🚀 Ejecutando script:', scriptPath);

    // Modo: predict (no entrenar), dataset: mnist, epocas: 0, sin recompilar
    const child = spawn('bash', [scriptPath, 'main', 'mnist', 'predict', '0', '--no-build'], {
      cwd: path.join(process.cwd(), '..'),
      env: {
        ...process.env,
        INPUT_IMAGE: imagePath  // <- imagen enviada por entorno
      }
    });

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
        console.error('❌ Script stderr:', stderr);
        return reject(new Error(`El script falló con código ${code}`));
      }

      try {
        const matches = stdout
          .split('\n')
          .filter(line => line.includes('Imagen predicción:'))
          .map(line => {
            const parts = line.split(':');
            return parts.length === 2 ? parseInt(parts[1].trim(), 10) : null;
          })
          .filter(val => val !== null);

        if (matches.length === 0) {
          throw new Error("No se encontraron predicciones válidas en la salida");
        }

        resolve({
          digits: matches,
          count: matches.length
        });
      } catch (err) {
        reject(new Error(`Error al parsear salida del script:\n${stdout}\n${err}`));
      }
    });

    child.on('error', (error) => {
      reject(new Error(`No se pudo ejecutar el script: ${error.message}`));
    });
  });
}
