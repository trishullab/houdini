# TestIntermediateTypeHandler

* `instantiateIntermediateTypes`: Given a program, instantiates sort variables with given concrete types..
* Works of multiple compositions.

    
Convention: Concrete sorts in small case, sort variables in capital
    
Example 1:
 
    input program: compose(f0: B --> c, f1: a --> B) 
    cmts = Concrete Midtypes = [x, y]
    
        [ compose(f0: B --> c, f1: a --> B) ]
    ~ { Instantiate B with x and then with y}
        [ compose(f0: x --> c, f1: a --> x),  compose(f0: y --> c, f1: a --> y)]
    
    
Example 2:
 
    input program: compose(compose(f0: B --> c, f1: a --> B),  compose(f0: C --> a, f1: d --> C)) 
    cmts = Concrete Midtypes = [x, y]
    
        [ compose(compose(f0: B --> c, f1: a --> B),  compose(f0: C --> a, f1: d --> C)) ]
    ~ { Instantiate B with x and then with y}
        [ compose(compose(f0: x --> c, f1: a --> x),  compose(f0: C --> a, f1: d --> C)),
          compose(compose(f0: y --> c, f1: a --> y),  compose(f0: C --> a, f1: d --> C)) ]
    ~ { Instantiate C with x and then with y}
        [ compose(compose(f0: x --> c, f1: a --> x),  compose(f0: x --> a, f1: d --> x)),
          compose(compose(f0: y --> c, f1: a --> y),  compose(f0: x --> a, f1: d --> x)),
          compose(compose(f0: x --> c, f1: a --> x),  compose(f0: y --> a, f1: d --> y)),
          compose(compose(f0: y --> c, f1: a --> y),  compose(f0: y --> a, f1: d --> y)) ]
          
Example in the test case:
    
    TODO



 
 