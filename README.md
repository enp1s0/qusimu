# SuperCon 2018 本戦問題 (量子計算機シミュレーション)

- 本戦の環境に合わせてGPU環境で動作させる

## 方針
### imp01
- 愚直に

### imp02
- カーネル関数の立ち上げは1回で済ませる
	- 入力ゲートとその引数を命令列に変換し，カーネル関数に渡す
	- カーネル関数内部でPCを持ち，命令列を処理する

## 最適化
- 命令列の圧縮
	- Xゲートとか同時に計算できそう

- レジスタをキャッシュとして用いる
	- Zの前とか結果をglobalに書き戻さなくてもいい

- 命令列をConstantメモリに乗せる
