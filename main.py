import subprocess
import os
import time

def run_mlflow_and_streamlit(mlflow_dir="mlflow", dashboard_file="dashboard.py"):
    try:
        if not os.path.exists(mlflow_dir):
            os.makedirs(mlflow_dir)
            print(f"Diretório '{mlflow_dir}' criado para o MLflow.")
        
        mlflow_process = subprocess.Popen(
            ["mlflow", "ui", "--backend-store-uri", mlflow_dir],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        print(f"Mlflow UI iniciado no diretório '{mlflow_dir}'. Acesse em http://127.0.0.1:5000")
        
        time.sleep(5)
        
        if not os.path.exists(dashboard_file):
            print(f"Arquivo '{dashboard_file}' não encontrado. Verifique o caminho.")
            mlflow_process.terminate()
            return
        
        streamlit_process = subprocess.Popen(
            ["streamlit", "run", dashboard_file],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        print("Streamlit iniciado. Acesse em http://127.0.0.1:8501")
        
        print("Pressione Ctrl+C para encerrar os serviços.")
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\nEncerrando os serviços...")
        # Finalizar os processos
        mlflow_process.terminate()
        streamlit_process.terminate()
        print("Serviços encerrados com sucesso.")
    except Exception as e:
        print(f"Erro: {e}")

if __name__ == "__main__":
    run_mlflow_and_streamlit(mlflow_dir="mlflow", dashboard_file="dashboard.py")
