@echo off
rem WINDOWS LAUNCH SCRIPT

rem ENVIRONMENT VARIABLES TO MODIFY

set JDK_PATH=%programfiles%\Java\jdk1.8.0_151
set WEKA=..\weka-3-4\weka.jar

rem ----------------------------------

set COMMONS_CLI=lib\commons-cli-1.0.jar
set JMRC=lib\jmrc.jar

set LIBS=%WEKA%;%COMMONS_CLI%;%JMRC%;%CD%;bin\

"C:\Program Files\Java\jdk-9.0.4\bin\java" -Xmx512m -classpath %LIBS% recognizer.PersonalityRecognizer %1 %2 %3 %4 %5 %6 %7 %8 %9
