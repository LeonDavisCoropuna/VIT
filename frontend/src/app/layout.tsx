import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'ViT Transformer Model Tester',
  description: 'Test your trained ViT transformer model with MNIST digits',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="es">
      <body>{children}</body>
    </html>
  )
}
