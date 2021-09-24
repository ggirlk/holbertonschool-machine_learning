-- comment

delimiter //
CREATE FUNCTION SafeDiv(a INTEGER, b INTEGER)
	RETURNS FLOAT
	BEGIN
		SET @result = 0;
		IF b <> 0 THEN
			SET @result = a/b;
		END IF;
		RETURN @result;
	END //
delimiter;
