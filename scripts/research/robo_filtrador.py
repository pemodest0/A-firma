#!/usr/bin/env python3
"""
Robô Filtrador de Arquivos CSV e Excel
Filtra dados e gera relatórios PDF automáticos
"""
import pandas as pd
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from datetime import datetime
import sys

def carregar_dados(caminho_arquivo):
    """Carrega dados de CSV ou Excel"""
    caminho = Path(caminho_arquivo)
    if caminho.suffix.lower() == '.csv':
        return pd.read_csv(caminho)
    elif caminho.suffix.lower() in ['.xlsx', '.xls']:
        return pd.read_excel(caminho)
    else:
        raise ValueError("Formato não suportado. Use CSV ou Excel.")

def filtrar_dados(df, filtros=None, remover_nan=True):
    """
    Filtra dados conforme critérios especificados
    
    Args:
        df: DataFrame com os dados
        filtros: Lista de dicionários com filtros
                 Ex: [{'coluna': 'valor', 'operador': '>', 'valor': 100}]
        remover_nan: Se True, remove linhas com NaN
    
    Returns:
        DataFrame filtrado e lista de erros
    """
    erros = []
    df_filtrado = df.copy()
    
    # Remover NaNs
    if remover_nan:
        linhas_antes = len(df_filtrado)
        df_filtrado = df_filtrado.dropna()
        removidas = linhas_antes - len(df_filtrado)
        if removidas > 0:
            erros.append(f"{removidas} linhas removidas por valores NaN")
    
    # Aplicar filtros
    if filtros:
        for filtro in filtros:
            coluna = filtro.get('coluna')
            operador = filtro.get('operador', '==')
            valor = filtro.get('valor')
            
            if coluna not in df_filtrado.columns:
                erros.append(f"Coluna '{coluna}' não encontrada")
                continue
            
            try:
                if operador == '>':
                    df_filtrado = df_filtrado[df_filtrado[coluna] > valor]
                elif operador == '<':
                    df_filtrado = df_filtrado[df_filtrado[coluna] < valor]
                elif operador == '>=':
                    df_filtrado = df_filtrado[df_filtrado[coluna] >= valor]
                elif operador == '<=':
                    df_filtrado = df_filtrado[df_filtrado[coluna] <= valor]
                elif operador == '==':
                    df_filtrado = df_filtrado[df_filtrado[coluna] == valor]
                elif operador == '!=':
                    df_filtrado = df_filtrado[df_filtrado[coluna] != valor]
            except Exception as e:
                erros.append(f"Erro ao aplicar filtro em '{coluna}': {e}")
    
    return df_filtrado, erros

def gerar_relatorio_pdf(df_original, df_filtrado, erros, caminho_saida):
    """Gera relatório PDF profissional"""
    doc = SimpleDocTemplate(str(caminho_saida), pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    # Título
    story.append(Paragraph("Relatório de Processamento de Dados", styles['Title']))
    story.append(Spacer(1, 12))
    
    # Informações gerais
    story.append(Paragraph(f"<b>Data de Processamento:</b> {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}", styles['Normal']))
    story.append(Spacer(1, 12))
    
    story.append(Paragraph("Resumo do Processamento:", styles['Heading2']))
    story.append(Paragraph(f"Linhas originais: {len(df_original)}", styles['Normal']))
    story.append(Paragraph(f"Linhas após filtragem: {len(df_filtrado)}", styles['Normal']))
    story.append(Paragraph(f"Linhas removidas: {len(df_original) - len(df_filtrado)}", styles['Normal']))
    story.append(Paragraph(f"Colunas: {len(df_original.columns)}", styles['Normal']))
    story.append(Spacer(1, 12))
    
    # Erros e avisos
    if erros:
        story.append(Paragraph("Erros e Avisos:", styles['Heading2']))
        for erro in erros:
            story.append(Paragraph(f"• {erro}", styles['Normal']))
        story.append(Spacer(1, 12))
    
    # Estatísticas
    story.append(Paragraph("Estatísticas dos Dados:", styles['Heading2']))
    for col in df_filtrado.select_dtypes(include=['number']).columns:
        story.append(Paragraph(f"<b>{col}:</b> Média={df_filtrado[col].mean():.2f}, Min={df_filtrado[col].min():.2f}, Max={df_filtrado[col].max():.2f}", styles['Normal']))
    story.append(Spacer(1, 12))
    
    # Tabela com dados (primeiras 50 linhas)
    story.append(Paragraph("Dados Processados (primeiras 50 linhas):", styles['Heading2']))
    data = [df_filtrado.columns.tolist()] + df_filtrado.head(50).values.tolist()
    table = Table(data)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 8),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTSIZE', (0, 1), (-1, -1), 7),
    ]))
    story.append(table)
    
    doc.build(story)
    print(f"Relatório PDF gerado: {caminho_saida}")

def main():
    # Configurações
    INPUT_FILE = 'templates_python/test_accounting_data.csv'  # Altere aqui
    OUTPUT_FILE = f'results/relatorio_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf'
    
    # Filtros (opcional)
    FILTROS = [
        # Exemplo: {'coluna': 'Debit', 'operador': '>', 'valor': 0}
    ]
    
    REMOVER_NAN = True
    
    try:
        # Carregar dados
        print(f"Carregando dados de {INPUT_FILE}...")
        df = carregar_dados(INPUT_FILE)
        print(f"Dados carregados: {df.shape[0]} linhas, {df.shape[1]} colunas")
        
        # Filtrar dados
        print("Filtrando dados...")
        df_filtrado, erros = filtrar_dados(df, FILTROS, REMOVER_NAN)
        print(f"Dados filtrados: {df_filtrado.shape[0]} linhas")
        
        if erros:
            print("Avisos:")
            for erro in erros:
                print(f"  - {erro}")
        
        # Criar diretório de saída
        Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)
        
        # Gerar relatório PDF
        print("Gerando relatório PDF...")
        gerar_relatorio_pdf(df, df_filtrado, erros, OUTPUT_FILE)
        
        print("Processo concluído com sucesso!")
        
    except Exception as e:
        print(f"Erro: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
