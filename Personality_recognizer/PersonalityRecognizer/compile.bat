@echo off
rem WINDOWS COMPILATION SCRIPT

rem ENVIRONMENT VARIABLES TO MODIFY

set JDK_PATH= %programfiles%\Java\jdk1.8.0_151
set WEKA= C:\Users\User\Documents\Python_materials\PersonalityRecognizer\weka-3-4\weka.jar

rem ----------------------------------

set COMMONS_CLI= C:\Users\User\Documents\Python_materials\PersonalityRecognizer\lib\commons-cli-1.0.jar

set LIBS=%WEKA%;%COMMONS_CLI%;%CD%;bin\

"C:\Program Files\Java\jdk1.8.0_151\bin\javac" -classpath %LIBS% src\recognizer\PersonalityRecognizer.java src\recognizer\Utils.java -d bin\