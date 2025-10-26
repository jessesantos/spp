import requests
import time
import json
from typing import Dict, Tuple
import logging
from pathlib import Path
from config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RateLimitError(Exception):
    """Exceção para rate limit da API"""
    pass


class SentimentAnalyzer:
    """Análise de sentimento usando Google Gemini API"""
    
    def __init__(self):
        self.api_url = f"{config.GEMINI_API_URL}?key={config.GEMINI_API_KEY}"
        self.headers = {"Content-Type": "application/json"}
        self.rate_limit_delay = 4.5  # 60s / 15 req = 4s, +0.5s margem de segurança
        self.cache_file = "sentiment_cache.json"
        self.cache = self._load_cache()
        self.max_retries = 3
    
    def _load_cache(self) -> Dict:
        """Carrega cache de sentimentos já analisados"""
        if Path(self.cache_file).exists():
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Erro ao carregar cache: {e}")
        return {}
    
    def _save_cache(self):
        """Salva cache de sentimentos"""
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"Erro ao salvar cache: {e}")
    
    def _build_prompt(self, text: str) -> str:
        """Constrói prompt para análise de sentimento"""
        return f"""Analise o sentimento da seguinte notícia financeira sobre PETR4/Petrobras e retorne APENAS um número:
-1 para sentimento NEGATIVO (notícia ruim para o preço da ação)
0 para sentimento NEUTRO (sem impacto claro)
1 para sentimento POSITIVO (notícia boa para o preço da ação)

Notícia: {text}

Retorne apenas o número (-1, 0 ou 1) sem explicações."""
    
    def analyze_sentiment(self, text: str) -> Tuple[int, float]:
        """
        Analisa sentimento de um texto usando Gemini
        
        Args:
            text: Texto da notícia para análise
            
        Returns:
            Tuple (sentiment_score, confidence) onde:
            - sentiment_score: -1 (negativo), 0 (neutro), 1 (positivo)
            - confidence: grau de confiança (0.0 a 1.0)
        """
        if not text or text.strip() == "" or text == "sem notícia relevante":
            return 0, 0.5  # neutro com baixa confiança
        
        # Verificar cache
        cache_key = text[:100]  # Usar primeiros 100 chars como chave
        if cache_key in self.cache:
            logger.info("Usando sentimento do cache")
            cached = self.cache[cache_key]
            return cached['sentiment'], cached['confidence']
        
        # Tentar com retry
        for attempt in range(self.max_retries):
            try:
                sentiment, confidence = self._call_api(text)
                
                # Salvar no cache
                self.cache[cache_key] = {
                    'sentiment': sentiment,
                    'confidence': confidence
                }
                self._save_cache()
                
                return sentiment, confidence
            
            except RateLimitError as e:
                if attempt < self.max_retries - 1:
                    wait_time = e.retry_after if hasattr(e, 'retry_after') else 60
                    logger.warning(f"Rate limit atingido. Aguardando {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    logger.error("Rate limit excedido após todas as tentativas")
                    return 0, 0.3
            
            except Exception as e:
                logger.error(f"Erro na tentativa {attempt + 1}: {str(e)}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)  # Backoff exponencial
                else:
                    return 0, 0.2
        
        return 0, 0.2
    
    def _call_api(self, text: str) -> Tuple[int, float]:
        """Faz chamada à API Gemini"""
        prompt = self._build_prompt(text)
        
        payload = {
            "contents": [{
                "parts": [{
                    "text": prompt
                }]
            }],
            "generationConfig": {
                "temperature": 0.1,
                "maxOutputTokens": 10
            }
        }
        
        response = requests.post(
            self.api_url,
            headers=self.headers,
            json=payload,
            timeout=30
        )
        
        if response.status_code == 429:
            # Rate limit excedido
            error_data = response.json()
            retry_after = 60  # default
            
            # Tentar extrair tempo de retry
            try:
                details = error_data.get('error', {}).get('details', [])
                for detail in details:
                    if detail.get('@type') == 'type.googleapis.com/google.rpc.RetryInfo':
                        retry_delay = detail.get('retryDelay', '60s')
                        retry_after = int(retry_delay.replace('s', ''))
                        break
            except:
                pass
            
            error = RateLimitError(f"Rate limit excedido")
            error.retry_after = retry_after
            raise error
        
        if response.status_code != 200:
            logger.warning(f"Erro na API Gemini: {response.status_code}")
            raise Exception(f"API error: {response.status_code}")
        
        result = response.json()
        
        # Extrair resposta
        if "candidates" in result and len(result["candidates"]) > 0:
            content = result["candidates"][0]["content"]
            text_response = content["parts"][0]["text"].strip()
            
            sentiment = self._parse_sentiment(text_response)
            confidence = 0.8
            
            # Respeitar rate limit
            time.sleep(self.rate_limit_delay)
            
            return sentiment, confidence
        
        raise Exception("Resposta inválida da API")
    
    def _parse_sentiment(self, text: str) -> int:
        """Parseia resposta do Gemini para extrair sentimento"""
        text = text.strip().lower()
        
        # Tentar extrair número direto
        if "-1" in text or "negativo" in text:
            return -1
        elif "1" in text and "-1" not in text or "positivo" in text:
            return 1
        else:
            return 0
    
    def batch_analyze(self, texts: list) -> Dict[int, Tuple[int, float]]:
        """
        Analisa sentimento de múltiplos textos em batch
        
        Args:
            texts: Lista de textos para análise
            
        Returns:
            Dicionário {index: (sentiment, confidence)}
        """
        results = {}
        total = len(texts)
        
        for idx, text in enumerate(texts):
            logger.info(f"Analisando sentimento {idx + 1}/{total}")
            sentiment, confidence = self.analyze_sentiment(text)
            results[idx] = (sentiment, confidence)
        
        return results