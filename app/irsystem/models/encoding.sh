#!/bin/bash
#enter input encoding here
FROM_ENCODING="value_here"
#output encoding(UTF-8)
TO_ENCODING="UTF-8"
#convert
CONVERT=" iconv  -f   $FROM_ENCODING  -t   $TO_ENCODING"
#loop to convert multiple files 
for  file  in  *.csv; do
     $CONVERT   "$file"   -o  "${file%.csv}.utf8.converted"
done
exit 0
