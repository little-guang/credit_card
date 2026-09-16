CREATE TABLE credit_card_zh AS
SELECT
    ID AS 客戶編號,
    LIMIT_BAL AS 信用額度,
    SEX AS 性別,
    EDUCATION AS 教育程度,
    MARRIAGE AS 婚姻狀況,
    AGE AS 年齡,

    PAY_0 AS 9月還款狀態,
    PAY_2 AS 8月還款狀態,
    PAY_3 AS 7月還款狀態,
    PAY_4 AS 6月還款狀態,
    PAY_5 AS 5月還款狀態,
    PAY_6 AS 4月還款狀態,

    BILL_AMT1 AS 9月帳單金額,
    BILL_AMT2 AS 8月帳單金額,
    BILL_AMT3 AS 7月帳單金額,
    BILL_AMT4 AS 6月帳單金額,
    BILL_AMT5 AS 5月帳單金額,
    BILL_AMT6 AS 4月帳單金額,

    PAY_AMT1 AS 9月還款金額,
    PAY_AMT2 AS 8月還款金額,
    PAY_AMT3 AS 7月還款金額,
    PAY_AMT4 AS 6月還款金額,
    PAY_AMT5 AS 5月還款金額,
    PAY_AMT6 AS 4月還款金額,

    default_payment_next_month AS 下月是否違約

FROM credit_card_raw;