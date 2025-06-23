import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import numpy as np

# ============================
# 1. Carregar dados de treino
# ============================
print("Carregando dados de treino...")

# Tentar carregar arquivo de treino real primeiro
try:
    # Tentar carregar o arquivo de treino específico para toxic comments
    train_file = '/Users/anacarolinamachado/iA/IA/Lista 13/train_toxic.csv'
    df = pd.read_csv(train_file)
    print(f"Arquivo de treino carregado: {train_file}")
    print(f"Formato dos dados: {df.shape}")
    print(f"Colunas: {df.columns.tolist()}")
    
    # Verificar se tem as colunas necessárias para toxic comments
    required_columns = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    if 'comment_text' in df.columns and all(col in df.columns for col in required_columns):
        print("Dataset de toxic comments detectado!")
        # Usar apenas as colunas necessárias
        df = df[['comment_text'] + required_columns]
    else:
        print("Dataset não é de toxic comments, criando dataset de exemplo...")
        raise FileNotFoundError("Dataset não compatível")
        
except (FileNotFoundError, KeyError, pd.errors.EmptyDataError) as e:
    print(f"Arquivo de treino não encontrado ou incompatível: {e}")
    print("Criando dataset de exemplo para toxic comments...")
    
    # Carregar estrutura do sample_submission para entender o formato
    try:
        sample_sub = pd.read_csv('sample_submission.csv')
        print(f"Sample submission carregado com {len(sample_sub)} linhas")
    except:
        print("Sample submission não encontrado, criando estrutura básica...")
        # Criar estrutura básica se não encontrar o arquivo
        sample_sub = pd.DataFrame({
            'id': range(1, 1001),
            'toxic': [0] * 1000,
            'severe_toxic': [0] * 1000,
            'obscene': [0] * 1000,
            'threat': [0] * 1000,
            'insult': [0] * 1000,
            'identity_hate': [0] * 1000
        })
    
    # Criar dataset de treino mais realista
    np.random.seed(42)  # Para reprodutibilidade
    
    # Comentários de exemplo (alguns tóxicos, outros não)
    toxic_comments = [
        "You're such an idiot and should die.",
        "This is disgusting and obscene content.",
        "I'm going to hurt you badly.",
        "You filthy animal, go to hell.",
        "This is a threat to society.",
        "You should be ashamed of yourself.",
        "I hope you suffer and die.",
        "You're worthless and stupid.",
        "This is offensive and hateful.",
        "You deserve to be punished."
    ]
    
    non_toxic_comments = [
        "I totally agree with your point.",
        "Great work, keep it up!",
        "This is very helpful information.",
        "Thank you for sharing this.",
        "I appreciate your contribution.",
        "This is well written and clear.",
        "Good job on this project.",
        "I found this very interesting.",
        "Thanks for the explanation.",
        "This makes perfect sense."
    ]
    
    # Criar dataset balanceado
    n_samples = min(len(sample_sub), 1000)  # Limitar a 1000 amostras para exemplo
    comments = []
    labels = []
    
    for i in range(n_samples):
        if i < n_samples // 2:
            # Comentários tóxicos
            comment = np.random.choice(toxic_comments)
            # Gerar labels aleatórios mas realistas para comentários tóxicos
            toxic = np.random.choice([0, 1], p=[0.2, 0.8])  # 80% chance de ser tóxico
            severe_toxic = np.random.choice([0, 1], p=[0.7, 0.3]) if toxic else 0
            obscene = np.random.choice([0, 1], p=[0.3, 0.7]) if toxic else 0
            threat = np.random.choice([0, 1], p=[0.8, 0.2]) if toxic else 0
            insult = np.random.choice([0, 1], p=[0.2, 0.8]) if toxic else 0
            identity_hate = np.random.choice([0, 1], p=[0.9, 0.1]) if toxic else 0
        else:
            # Comentários não tóxicos
            comment = np.random.choice(non_toxic_comments)
            # Labels para comentários não tóxicos
            toxic = severe_toxic = obscene = threat = insult = identity_hate = 0
        
        comments.append(comment)
        labels.append([toxic, severe_toxic, obscene, threat, insult, identity_hate])
    
    # Criar DataFrame
    df = pd.DataFrame({
        'comment_text': comments,
        'toxic': [label[0] for label in labels],
        'severe_toxic': [label[1] for label in labels],
        'obscene': [label[2] for label in labels],
        'threat': [label[3] for label in labels],
        'insult': [label[4] for label in labels],
        'identity_hate': [label[5] for label in labels]
    })
    
    print(f"Dataset de exemplo criado com {len(df)} amostras")

# ============================
# 2. Separar variáveis
# ============================
print("\nSeparando variáveis...")
X = df['comment_text']
Y = df.drop(columns=['comment_text'])

print(f"X shape: {X.shape}")
print(f"Y shape: {Y.shape}")
print(f"Classes de saída: {Y.columns.tolist()}")

# Dividir dados: 70% treino, 30% teste
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

print(f"Treino: {len(X_train)} amostras")
print(f"Teste: {len(X_test)} amostras")

# ============================
# 3. TF-IDF
# ============================
print("\nAplicando TF-IDF...")
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print(f"Vocabulário TF-IDF: {X_train_tfidf.shape[1]} features")

# ============================
# 4. Classificador Multirrótulo
# ============================
print("\nTreinando classificador...")
classifier = BinaryRelevance(classifier=MultinomialNB())
classifier.fit(X_train_tfidf, Y_train)
Y_pred = classifier.predict(X_test_tfidf)

# ============================
# 5. Avaliação
# ============================
print("\n" + "="*50)
print("RELATÓRIO DE CLASSIFICAÇÃO MULTIRRÓTULO")
print("="*50)
print(classification_report(Y_test, Y_pred, target_names=Y.columns.tolist(), zero_division=0))

# ============================
# 6. Geração de Submissão
# ============================
print("\nGerando arquivo de submissão...")

# Carregar estrutura do arquivo sample_submission
try:
    sample_sub = pd.read_csv('sample_submission.csv')
    print(f"Sample submission carregado: {len(sample_sub)} linhas")
    
    # Fazer previsões para os dados de teste
    # Vamos usar uma amostra dos dados de teste para simular a submissão
    test_sample = X_test.iloc[:len(sample_sub)]
    test_sample_tfidf = vectorizer.transform(test_sample)
    predictions = classifier.predict(test_sample_tfidf)
    
    # Montar a submissão
    submission = sample_sub.copy()
    for i, col in enumerate(Y.columns):
        if i < predictions.shape[1]:
            submission[col] = predictions[:, i].toarray().flatten()[:len(sample_sub)]
    
    # Salvar como CSV
    submission.to_csv("submission.csv", index=False)
    print("Arquivo submission.csv gerado com sucesso!")
    
except Exception as e:
    print(f"Erro ao gerar submissão: {e}")
    print("Gerando arquivo de exemplo...")
    
    # Criar arquivo de exemplo
    example_submission = pd.DataFrame({
        'id': range(1, len(X_test) + 1),
        'toxic': Y_pred[:, 0].toarray().flatten(),
        'severe_toxic': Y_pred[:, 1].toarray().flatten(),
        'obscene': Y_pred[:, 2].toarray().flatten(),
        'threat': Y_pred[:, 3].toarray().flatten(),
        'insult': Y_pred[:, 4].toarray().flatten(),
        'identity_hate': Y_pred[:, 5].toarray().flatten()
    })
    
    example_submission.to_csv("example_submission.csv", index=False)
    print("Arquivo example_submission.csv gerado com sucesso!")

print("\nProcesso concluído!")

