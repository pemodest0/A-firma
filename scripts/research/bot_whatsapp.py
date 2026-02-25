#!/usr/bin/env python3
"""
Bot Inteligente para WhatsApp
Responde automaticamente de acordo com os interesses da empresa
"""
import json
from datetime import datetime
from typing import Dict, List, Optional

class WhatsAppBot:
    """Bot inteligente para WhatsApp"""
    
    def __init__(self, empresa_config: Optional[Dict] = None):
        """
        Inicializa o bot com configuração da empresa
        
        Args:
            empresa_config: Dicionário com configurações da empresa
                          Ex: {'nome': 'Empresa XYZ', 'setor': 'vendas', ...}
        """
        self.config = empresa_config or {}
        self.historico = []
        self.respostas_base = self._carregar_respostas_base()
        self.respostas_personalizadas = self._carregar_respostas_personalizadas()
    
    def _carregar_respostas_base(self) -> Dict[str, str]:
        """Carrega respostas base do bot"""
        return {
            "saudacao": [
                "Olá! Como posso ajudar você hoje?",
                "Oi! Em que posso ajudar?",
                "Bem-vindo! Como posso ajudar?"
            ],
            "horario": [
                "Funcionamos de segunda a sexta, das 8h às 18h. Aos sábados das 9h às 13h.",
                "Nosso horário de atendimento é de segunda a sexta, 8h às 18h."
            ],
            "preco": [
                "Temos diferentes planos disponíveis. O plano básico custa R$ 299/mês. Para empresas maiores, oferecemos planos personalizados.",
                "Nossos preços variam conforme o plano. Gostaria de agendar uma demonstração para ver qual plano se encaixa melhor?"
            ],
            "contato": [
                "Você pode nos contatar por email (contato@assyntrax.com) ou WhatsApp. Nossa equipe está pronta para ajudar!",
                "Entre em contato conosco por email ou WhatsApp. Respondemos em até 2 horas úteis."
            ],
            "produto": [
                "Oferecemos soluções de automação, análise de dados e bots inteligentes. Qual área te interessa mais?",
                "Nossos principais produtos são: Robô Filtrador de Arquivos, Bot WhatsApp e Aplicações Web. Quer saber mais sobre algum?"
            ],
            "despedida": [
                "Foi um prazer ajudar! Se precisar de mais alguma coisa, estou aqui!",
                "Até logo! Qualquer dúvida, pode me chamar.",
                "Tchau! Espero ter ajudado."
            ]
        }
    
    def _carregar_respostas_personalizadas(self) -> Dict[str, str]:
        """Carrega respostas personalizadas baseadas na configuração da empresa"""
        setor = self.config.get('setor', '').lower()
        
        personalizacoes = {
            'vendas': {
                "produto": "Nossas soluções ajudam a automatizar processos de vendas, gerar relatórios de performance e otimizar o atendimento ao cliente.",
                "preco": "Para equipes de vendas, oferecemos planos a partir de R$ 499/mês com recursos avançados de CRM."
            },
            'rh': {
                "produto": "Automatizamos processos de RH como análise de dados de funcionários, geração de relatórios e gestão de informações.",
                "preco": "Planos para RH começam em R$ 399/mês com foco em automação de processos administrativos."
            },
            'financeiro': {
                "produto": "Nossas soluções financeiras incluem processamento automático de planilhas, geração de relatórios e análise de dados contábeis.",
                "preco": "Para departamentos financeiros, temos planos a partir de R$ 599/mês com recursos de segurança avançados."
            }
        }
        
        return personalizacoes.get(setor, {})
    
    def processar_mensagem(self, mensagem: str, contexto: Optional[Dict] = None) -> Dict:
        """
        Processa uma mensagem e retorna resposta
        
        Args:
            mensagem: Mensagem recebida
            contexto: Contexto adicional da conversa
        
        Returns:
            Dicionário com resposta, confiança e informações adicionais
        """
        mensagem_lower = mensagem.lower()
        contexto = contexto or {}
        
        # Detectar intenção
        intencao = self._detectar_intencao(mensagem_lower)
        
        # Gerar resposta
        resposta, confianca = self._gerar_resposta(intencao, mensagem_lower, contexto)
        
        # Salvar no histórico
        self.historico.append({
            "timestamp": datetime.now().isoformat(),
            "mensagem": mensagem,
            "intencao": intencao,
            "resposta": resposta,
            "confianca": confianca
        })
        
        return {
            "resposta": resposta,
            "intencao": intencao,
            "confianca": confianca,
            "timestamp": datetime.now().isoformat()
        }
    
    def _detectar_intencao(self, mensagem: str) -> str:
        """Detecta a intenção da mensagem"""
        palavras_chave = {
            "saudacao": ["ola", "oi", "bom dia", "boa tarde", "boa noite", "hey"],
            "horario": ["horario", "horário", "funciona", "aberto", "atendimento"],
            "preco": ["preco", "preço", "custo", "valor", "quanto", "price"],
            "contato": ["contato", "telefone", "email", "falar", "conversar"],
            "produto": ["produto", "servico", "serviço", "solucao", "solução", "o que fazem"],
            "despedida": ["tchau", "ate logo", "obrigado", "obrigada", "valeu"]
        }
        
        for intencao, palavras in palavras_chave.items():
            if any(palavra in mensagem for palavra in palavras):
                return intencao
        
        return "geral"
    
    def _gerar_resposta(self, intencao: str, mensagem: str, contexto: Dict) -> tuple:
        """Gera resposta baseada na intenção"""
        import random
        
        # Tentar resposta personalizada primeiro
        if intencao in self.respostas_personalizadas:
            resposta = self.respostas_personalizadas[intencao]
            confianca = 0.9
            return resposta, confianca
        
        # Usar respostas base
        if intencao in self.respostas_base:
            respostas = self.respostas_base[intencao]
            resposta = random.choice(respostas)
            confianca = 0.8
            return resposta, confianca
        
        # Resposta padrão
        respostas_padrao = [
            "Obrigado pela sua mensagem! Nossa equipe está analisando sua solicitação e retornará em breve.",
            "Entendi sua mensagem. Posso ajudar com informações sobre horários, preços, produtos ou contato. O que você gostaria de saber?",
            "Como posso ajudar você hoje? Posso fornecer informações sobre nossos produtos, preços, horários de atendimento ou outras dúvidas."
        ]
        resposta = random.choice(respostas_padrao)
        confianca = 0.5
        
        return resposta, confianca
    
    def obter_historico(self) -> List[Dict]:
        """Retorna histórico de conversas"""
        return self.historico
    
    def salvar_historico(self, caminho: str):
        """Salva histórico em arquivo JSON"""
        with open(caminho, 'w', encoding='utf-8') as f:
            json.dump(self.historico, f, ensure_ascii=False, indent=2)
        print(f"Histórico salvo em {caminho}")

def main():
    """Exemplo de uso do bot"""
    # Configurar bot para uma empresa
    config_empresa = {
        "nome": "Empresa XYZ",
        "setor": "vendas",
        "contato": "contato@empresa.com"
    }
    
    bot = WhatsAppBot(config_empresa)
    
    # Simular conversas
    mensagens_teste = [
        "Olá, bom dia!",
        "Quais são os horários de funcionamento?",
        "Quanto custa o produto?",
        "Como posso entrar em contato?",
        "Obrigado, tchau!"
    ]
    
    print("=== Simulação de Conversas com Bot WhatsApp ===\n")
    
    for mensagem in mensagens_teste:
        print(f"Usuário: {mensagem}")
        resultado = bot.processar_mensagem(mensagem)
        print(f"Bot: {resultado['resposta']}")
        print(f"Intenção: {resultado['intencao']} | Confiança: {resultado['confianca']:.2f}\n")
    
    # Salvar histórico
    bot.salvar_historico("results/historico_bot.json")
    print("Histórico salvo!")

if __name__ == "__main__":
    main()
