-- comment

delimiter //
CREATE TRIGGER update_email
	BEFORE UPDATE
	ON users
	FOR EACH ROW
	BEGIN
  		IF STRCMP(old.email, new.email) <> 0 THEN
			SET new.valid_email = 0;
		END IF;
	END //
delimiter;
