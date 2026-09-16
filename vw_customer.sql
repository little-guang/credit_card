CREATE VIEW vw_customer AS
SELECT
    c.ID AS '客戶編號',
    c.LIMIT_BAL AS '信用額度',
    s.SEX_Name AS '性別',
    e.EDUCATION_Name AS '教育程度',
    m.MARRIAGE_Name AS '婚姻狀況',
    c.AGE AS '年紀'
FROM credit_card_raw AS c
LEFT JOIN SEX AS s
    ON c.SEX = s.SEX
LEFT JOIN education AS e
    ON c.EDUCATION = e.EDUCATION
LEFT JOIN marital_status AS m
    ON c.MARRIAGE = m.MARRIAGE;