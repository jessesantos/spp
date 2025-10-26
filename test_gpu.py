#!/usr/bin/env python3
"""
Script para detectar, listar e testar GPUs disponíveis
"""

import sys
from gpu_manager import GPUManager, setup_gpu
import tensorflow as tf


def main():
    print("\n" + "=" * 70)
    print("  🎮 DETECÇÃO E TESTE DE GPU")
    print("=" * 70)
    
    # Criar gerenciador
    manager = GPUManager()
    
    # Listar GPUs
    manager.print_gpu_info()
    
    # Informações do TensorFlow
    print("\n📊 INFORMAÇÕES DO TENSORFLOW")
    print("-" * 70)
    print(f"Versão: {tf.__version__}")
    print(f"Built with CUDA: {tf.test.is_built_with_cuda()}")
    print(f"GPU disponível: {tf.test.is_gpu_available()}")
    
    # Testar cada GPU
    gpus = manager.list_gpus()
    
    if not gpus:
        print("\n" + "=" * 70)
        print("❌ NENHUMA GPU DETECTADA")
        print("=" * 70)
        print("\n🔧 DIAGNÓSTICO:")
        print("-" * 70)
        print("1. Verificar drivers NVIDIA:")
        print("   $ nvidia-smi")
        print("\n2. Verificar CUDA Toolkit:")
        print("   $ nvcc --version")
        print("\n3. Verificar TensorFlow-GPU:")
        print("   $ python -c 'import tensorflow as tf; print(tf.config.list_physical_devices())'")
        print("\n4. Instalar CUDA e cuDNN:")
        print("   https://www.tensorflow.org/install/gpu")
        print("=" * 70 + "\n")
        return 1
    
    print("\n🧪 TESTE DE GPUs")
    print("-" * 70)
    
    for gpu in gpus:
        gpu_id = gpu['id']
        print(f"\nTestando GPU {gpu_id}...")
        
        try:
            # Configurar GPU
            test_manager = setup_gpu(gpu_id=gpu_id, memory_growth=True)
            
            # Operação simples
            with tf.device(f'/GPU:{gpu_id}'):
                a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
                b = tf.constant([[5.0, 6.0], [7.0, 8.0]])
                c = tf.matmul(a, b)
            
            print(f"✅ GPU {gpu_id}: OK")
            
        except Exception as e:
            print(f"❌ GPU {gpu_id}: FALHOU - {str(e)}")
    
    # Menu interativo
    print("\n" + "=" * 70)
    print("⚙️  CONFIGURAR GPU")
    print("=" * 70)
    print("\nEdite config.py para selecionar GPU:")
    print(f"\n# Usar GPU específica")
    print(f"GPU_ID: int = 0  # 0, 1, 2, ...")
    print(f"\n# Usar todas as GPUs")
    print(f"GPU_ID: int = None")
    print(f"\n# Usar apenas CPU")
    print(f"GPU_ID: int = -1")
    print("\n" + "=" * 70)
    
    # Teste com escolha do usuário
    if len(gpus) > 1:
        print("\n🎯 TESTE RÁPIDO")
        print("-" * 70)
        
        try:
            choice = input(f"\nEscolha GPU para testar (0-{len(gpus)-1}, Enter=pular): ").strip()
            
            if choice:
                gpu_id = int(choice)
                
                if 0 <= gpu_id < len(gpus):
                    print(f"\nTestando GPU {gpu_id}...")
                    manager.select_gpu(gpu_id)
                    
                    # Operação mais pesada
                    with tf.device(f'/GPU:{gpu_id}'):
                        matrix = tf.random.normal([1000, 1000])
                        result = tf.matmul(matrix, matrix)
                    
                    print(f"✅ GPU {gpu_id} funcionando corretamente!")
                else:
                    print("❌ ID inválido")
        
        except KeyboardInterrupt:
            print("\n\n⚠️  Cancelado pelo usuário")
        except Exception as e:
            print(f"\n❌ Erro: {e}")
    
    print("\n" + "=" * 70)
    print("✅ DIAGNÓSTICO COMPLETO")
    print("=" * 70 + "\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
