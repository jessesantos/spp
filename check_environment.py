#!/usr/bin/env python3
"""
Script de verificação de ambiente e dependências
Verifica se o sistema está pronto para executar
"""

import sys
import subprocess
from pathlib import Path


def print_header(text: str):
    """Imprime cabeçalho formatado"""
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)


def print_check(condition: bool, message: str):
    """Imprime resultado de verificação"""
    status = "✅" if condition else "❌"
    print(f"{status} {message}")
    return condition


def check_python_version():
    """Verifica versão do Python"""
    print_header("🐍 Verificação de Python")
    
    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"
    
    print(f"Versão instalada: Python {version_str}")
    
    is_valid = version.major == 3 and version.minor >= 12
    
    if is_valid:
        print_check(True, "Python 3.12+ detectado")
    else:
        print_check(False, f"Python 3.12+ necessário (encontrado: {version_str})")
        print("\n💡 Como atualizar:")
        print("   - Ubuntu/Debian: sudo apt install python3.12")
        print("   - macOS: brew install python@3.12")
        print("   - Windows: https://www.python.org/downloads/")
    
    return is_valid


def check_venv():
    """Verifica se está em ambiente virtual"""
    print_header("📦 Verificação de Ambiente Virtual")
    
    in_venv = sys.prefix != sys.base_prefix
    
    if in_venv:
        print_check(True, "Ambiente virtual ativo")
        print(f"   Localização: {sys.prefix}")
    else:
        print_check(False, "Ambiente virtual não detectado")
        print("\n💡 Como criar:")
        print("   python -m venv venv")
        print("   source venv/bin/activate  # Linux/Mac")
        print("   venv\\Scripts\\activate    # Windows")
    
    return in_venv


def check_dependencies():
    """Verifica dependências instaladas"""
    print_header("📚 Verificação de Dependências")
    
    dependencies = {
        "pandas": ">=2.2.0",
        "numpy": ">=1.26.0",
        "tensorflow": ">=2.16.1",
        "sklearn": ">=1.4.0",
        "requests": ">=2.31.0",
        "dotenv": ">=1.0.0"
    }
    
    all_ok = True
    
    for package, min_version in dependencies.items():
        try:
            if package == "sklearn":
                # scikit-learn importa como sklearn
                import sklearn
                version = sklearn.__version__
                package_name = "scikit-learn"
            elif package == "dotenv":
                # python-dotenv importa como dotenv
                import dotenv
                version = dotenv.__version__
                package_name = "python-dotenv"
            else:
                module = __import__(package)
                version = module.__version__
                package_name = package
            
            print_check(True, f"{package_name} {version}")
        
        except ImportError:
            print_check(False, f"{package_name} não instalado")
            all_ok = False
        
        except AttributeError:
            print_check(True, f"{package_name} instalado (versão não detectada)")
    
    if not all_ok:
        print("\n💡 Instalar dependências:")
        print("   pip install -r requirements.txt")
    
    return all_ok


def check_files():
    """Verifica arquivos necessários"""
    print_header("📄 Verificação de Arquivos")
    
    required_files = {
        ".env": "Arquivo de configuração (API keys)",
        "PETR4_2025_10_01_to_2025_10_25.csv": "Dados históricos",
        "noticias_PETR4_setembro_2025.csv": "Dados de notícias"
    }
    
    all_ok = True
    
    for filename, description in required_files.items():
        exists = Path(filename).exists()
        
        if filename == ".env":
            # .env é crítico mas pode ser criado
            if not exists:
                print_check(False, f"{filename} - {description}")
                print("   💡 Criar arquivo .env com: GEMINI_API_KEY=sua_chave")
                all_ok = False
            else:
                # Verificar se tem conteúdo
                content = Path(filename).read_text().strip()
                if "GEMINI_API_KEY" in content and len(content) > 20:
                    print_check(True, f"{filename} - {description}")
                else:
                    print_check(False, f"{filename} existe mas parece vazio")
                    all_ok = False
        else:
            print_check(exists, f"{filename} - {description}")
            if not exists:
                all_ok = False
    
    return all_ok


def check_gpu():
    """Verifica disponibilidade de GPU (opcional)"""
    print_header("🎮 Verificação de GPU (opcional)")
    
    try:
        import tensorflow as tf
        from gpu_manager import GPUManager
        
        manager = GPUManager()
        gpus = manager.list_gpus()
        
        if gpus:
            print_check(True, f"{len(gpus)} GPU(s) detectada(s)")
            for gpu in gpus:
                print(f"   [{gpu['id']}] {gpu['name']}")
            
            print("\n💡 Para selecionar GPU:")
            print("   1. Edite config.py:")
            print("      GPU_ID = 0  # Para GPU específica")
            print("      GPU_ID = None  # Para todas")
            print("      GPU_ID = -1  # Para CPU apenas")
            print("\n   2. Ou execute: python test_gpu.py")
        else:
            print_check(True, "Nenhuma GPU detectada (CPU será usado)")
            print("   ℹ️  GPU não é obrigatório, mas acelera o treinamento")
            print("\n   🔧 Para habilitar GPU:")
            print("   1. Instale drivers NVIDIA: nvidia-smi")
            print("   2. Instale CUDA Toolkit: https://developer.nvidia.com/cuda-downloads")
            print("   3. Execute: python test_gpu.py")
    
    except Exception as e:
        print_check(True, "Verificação de GPU pulada")
        print(f"   ℹ️  {str(e)}")
    
    return True  # GPU é opcional


def main():
    """Função principal"""
    print("""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║     🧠 SISTEMA DE PREDIÇÃO PETR4                            ║
║     Verificador de Ambiente                                 ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    checks = [
        ("Python 3.12+", check_python_version),
        ("Ambiente Virtual", check_venv),
        ("Dependências", check_dependencies),
        ("Arquivos", check_files),
        ("GPU", check_gpu)
    ]
    
    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ Erro ao verificar {name}: {e}")
            results.append((name, False))
    
    # Resumo final
    print_header("📊 Resumo Final")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ OK" if result else "❌ FALHOU"
        print(f"{status:12} {name}")
    
    print("\n" + "-" * 60)
    print(f"Total: {passed}/{total} verificações passaram")
    
    if passed == total:
        print("\n" + "🎉" * 30)
        print("\n✅ AMBIENTE PRONTO!")
        print("\nPróximos passos:")
        print("   1. python main.py              # Executar pipeline completo")
        print("   2. python predictor.py --days 5  # Fazer predições")
        print("\n" + "🎉" * 30)
        return 0
    
    else:
        print("\n⚠️  ATENÇÃO: Algumas verificações falharam")
        print("\nCorreções necessárias listadas acima.")
        print("Após corrigir, execute este script novamente.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
