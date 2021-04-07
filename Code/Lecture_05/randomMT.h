typedef unsigned long uint32;

class randMT {
    
    static const int N =          624;                // length of state vector
    static const int M =          397;                // a period parameter
    static const uint32 K =       0x9908B0DFU;        // a magic constant
    
    // If you want a single generator, consider using a singleton class
    // instead of trying to make these static.
    uint32   state[N+1];  // state vector + 1 extra to not violate ANSI C
    uint32   *next;       // next random value is computed from here
    uint32   initseed;    //
    int      left;        // can *next++ this many times before reloading
    
    inline uint32 hiBit(uint32 u) {
        return u & 0x80000000U;    // mask all but highest   bit of u
    }
    
    inline uint32 loBit(uint32 u) {
        return u & 0x00000001U;    // mask all but lowest    bit of u
    }
    
    inline uint32 loBits(uint32 u) {
        return u & 0x7FFFFFFFU;   // mask     the highest   bit of u
    }
    
    inline uint32 mixBits(uint32 u, uint32 v) {
        return hiBit(u)|loBits(v);  // move hi bit of u to hi bit of v
    }
    
    uint32 reload(void) ;
    
public:
    randMT() ;
    randMT(uint32 seed_) ;
    
    inline uint32 random(void) ;
    inline double rand() ;
    inline double rand( const double& n ) ;
    inline double randExc() ;
    inline double randExc( const double& n ) ;
    inline uint32 randInt( const uint32& n ) ;
    
    void seed(uint32 seed_) ;
};

randMT::randMT() {
    seed(1U);
}

randMT::randMT(uint32 seed_) {
    seed(seed_);
}

void randMT::seed(uint32 seed_) {
    initseed = seed_;
    
    uint32 x = (seed_ | 1U) & 0xFFFFFFFFU, *s = state;
    int j;
    left = 0;
    for(*s++=x, j=N; --j; *s++ = (x*=69069U) & 0xFFFFFFFFU);
}

uint32 randMT::reload(void) {
    uint32 *p0=state, *p2=state+2, *pM=state+M, s0, s1;
    int j;
    
    if(left < -1)
        seed(initseed);
    
    left=N-1, next=state+1;
    
    for(s0=state[0], s1=state[1], j=N-M+1; --j; s0=s1, s1=*p2++)
        *p0++ = *pM++ ^ (mixBits(s0, s1) >> 1) ^ (loBit(s1) ? K : 0U);
    
    for(pM=state, j=M; --j; s0=s1, s1=*p2++)
        *p0++ = *pM++ ^ (mixBits(s0, s1) >> 1) ^ (loBit(s1) ? K : 0U);
    
    s1=state[0], *p0 = *pM ^ (mixBits(s0, s1) >> 1) ^ (loBit(s1) ? K : 0U);
    s1 ^= (s1 >> 11);
    s1 ^= (s1 <<  7) & 0x9D2C5680U;
    s1 ^= (s1 << 15) & 0xEFC60000U;
    return(s1 ^ (s1 >> 18));
}

inline uint32 randMT::random(void) {
    uint32 y;
    
    if(--left < 0)
        return(reload());
    
    y  = *next++;
    y ^= (y >> 11);
    y ^= (y <<  7) & 0x9D2C5680U;
    y ^= (y << 15) & 0xEFC60000U;
    return(y ^ (y >> 18));
}

inline double randMT::rand()
{ return double(random()) * 2.3283064370807974e-10; }

inline double randMT::rand( const double& n )
{ return rand() * n; }

inline double randMT::randExc()
{ return double(random()) * 2.3283064365386963e-10; }

inline double randMT::randExc( const double& n )
{ return randExc() * n; }

inline uint32 randMT::randInt( const uint32& n )
{ return int( randExc() * (n+1) ); }
