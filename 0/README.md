# 1
Der Quellcode liest das Bild mit Hilfe der Image Klasse vom PIL Package ein. Dies wird dann mit Hilfe von Numpy als Array von unsigned Bytes gespeichert und durch Matplotlib dargestellt (geplottet)
würde man den dType der np-Methode ändern, könnte das Bild nicht mehr eingelesen werden.

# 2
das eingelesene Bild wird vervielfacht auf der x Achse, und dann auf der X-Achse, abhängig von den n und m Parametern (n=anzahl X Wiederholung, m=Anzahl Y Wiederholung)

# 3
ein Teil des eingelesenen Bildes wird ausgeschnitten und dargestellt. Der Ausschnitt ist hierbei [X1:X2, Y1:Y2] als Koordinaten des Ausschnitts.
Der Abstand zwischen X1 und X2 bzw Y1 und Y2 darf nicht 0 oder kleiner sein.