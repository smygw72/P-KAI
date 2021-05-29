## コンポーネント設計

### HomeView
- Header
- Footer
- Title（タイトル文字とタイトル画像？）
- MainMenu
    - ToHelpBtn（使い方ページへの遷移）
    - ToAsmtViewBtn（採点登録ページへの遷移）
***
### HeaderView
- ToHomeViewBtn
- ToAsmtViewBtn
- ToRecordViewBtn
- ToRankingViewBtn
- ToLoginLogoutView
***
### FooterView
- 利用規約
- お問い合わせ
- プライバシー/法的事項
- 開発計画
- Our Profile（TwitterとかQiitaとか）
- 著作権マーク
- 退会
***
### AsmtView
- 採点入力画面
    - タグ表示画面
        - タグ選択ボタン
            - タグ詳細選択画面
                - タグ検索（登録）入力フォーム
                - タグ一覧表示
                - 新規タグ登録ボタン
        - 選択タグの表示画面
    - ファイル選択画面
        - ファイル選択ボタン
        - 選択ファイル表示画面
    - ファイル名入力画面
        - ファイル名入力フォーム
    - 採点ボタン

- 点数表示画面
    - 点数表示画面
    - ランキング表示画面

- 登録画面
    - Recordへ登録ボタン
    - Rankingへ登録ボタン
***
### RecordView
- タグ表示画面
- 記録表示画面
***
### RankingView
- タグ表示画面
- 記録表示画面