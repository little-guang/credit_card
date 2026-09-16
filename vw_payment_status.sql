CREATE OR REPLACE VIEW vw_payment_status AS
SELECT
    c.ID AS '客戶編號',
    
    c.PAY_0 AS '9月還款狀態',
    c.PAY_2 AS '8月還款狀態',
    c.PAY_3 AS '7月還款狀態',
    c.PAY_4 AS '6月還款狀態',
    c.PAY_5 AS '5月還款狀態',
    c.PAY_6 AS '4月還款狀態'
FROM credit_card_raw AS c;
