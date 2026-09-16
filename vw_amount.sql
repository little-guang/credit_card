CREATE OR REPLACE VIEW vw_amount AS
SELECT
    c.ID AS '客戶編號',
    
    c.BILL_AMT1 AS '9月帳單金額',
    c.BILL_AMT2 AS '8月帳單金額',
    c.BILL_AMT3 AS '7月帳單金額',
    c.BILL_AMT4 AS '6月帳單金額',
    c.BILL_AMT5 AS '5月帳單金額',
    c.BILL_AMT6 AS '4月帳單金額',

    c.PAY_AMT1 AS '9月還款金額',
    c.PAY_AMT2 AS '8月還款金額',
    c.PAY_AMT3 AS '7月還款金額',
    c.PAY_AMT4 AS '6月還款金額',
    c.PAY_AMT5 AS '5月還款金額',
    c.PAY_AMT6 AS '4月還款金額'
FROM credit_card_raw AS c;