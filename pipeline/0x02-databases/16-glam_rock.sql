-- comment

SELECT band_name , IF(split IS NULL, (2020-formed), (split - formed)) AS lifespan FROM metal_bands
	WHERE style like '%Glam Rock%'
	ORDER BY lifespan DESC;
