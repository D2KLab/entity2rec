LibraryThing (LT) data 

Statistics:
Tag assignments: 2,056,487
Ratings:         749,401
Unique users:    7279
Unique items:    37,232
Unique tags:     10,559


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

UI2.txt 
Contains the rating and tagging information for each user

Format:
U_ID Work_ID 2*Rating Tag\n

U_ID:     The user identifyer.
Work_ID:  The work identifyer.
          More info: http://www.librarything.com/work/*Work_ID*
2*Rating: The user's rating for this book times two.
          This avoids floats as the ratings in LT can be given on a scale with steps of 0.5.
Tag:      A tag that was assigned by this user to this work.

Size:
2,056,487 lines

Comments:
Each User-Item-Rating combination occurs n times if a user has given n tags to that item.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Friends.txt
Contains a user's name and his friends, see http://www.librarything.com/wiki/index.php/Your_profile

Format:
U_ID U_Name F1 F2 ... 

U_ID:     The user identifyer.
U_Name:   The user's name.
          More info: http://www.librarything.com/catalog/*USER*
Fx:       The identifyer of this user's friend (if he has any)

Size:
7,279 lines

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Interesting.txt
Contains a user's name and the users this user considers interesting. see http://www.librarything.com/wiki/index.php/Your_profile

Format:
U_ID U_Name I1 I2 ... 

U_ID:     The user identifyer.
U_Name:   The user's name.
          More info: http://www.librarything.com/catalog/*USER*
Ix:       The identifyer of this user's interesting relations (if he has any)

Size:
7,279 lines