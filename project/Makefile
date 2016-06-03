CC=g++
CFLAGS=-c -Wall
LDFLAGS=
SOURCES=main.cpp deconvolution.cpp matrix_utility.cpp
OBJECTS=$(SOURCES:.cpp=.o)
EXECUTABLE=deblur

all: $(SOURCES) $(EXECUTABLE)
    
$(EXECUTABLE): $(OBJECTS) 
	$(CC) $(LDFLAGS) $(OBJECTS) -o $@

.cpp.o:
	$(CC) $(CFLAGS) $< -o $@

clean : 
	-rm -rf *.o
	-rm -rf $(EXECUTABLE)