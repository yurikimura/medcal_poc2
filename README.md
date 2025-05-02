### 起動

イメージをビルド
```
docker build -t medical-diagnosis-bot .
```

Dockerコンテナを実行
```
docker run -p 8501:8501 medical-diagnosis-bot
```

`http://localhost:8501` にアクセス
