<html>
<head>
<title>Capture the flag</title>
</head>
<body>
<h1 style='color:red'>Capture the flag</h1>
<form method='get' action='//127.0.0.1:8080/RCE2.php'>
<label>enter tour name</label>
<input type='text' name='name'><br>
<label>show time</label>
<input type='checkbox' name='date' value='date'><br>
<input type='submit'>
</form>
</body>
</html>
<?php
$cmd = $_GET['date'];
$name = $_GET['name'];
system($cmd);
echo  $name;
?>
