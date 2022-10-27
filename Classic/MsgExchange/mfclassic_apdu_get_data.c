// INCLUDE
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif // HAVE_CONFIG_H
#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <unistd.h>
#include <assert.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <sys/time.h>

// NFC
#include <nfc/nfc.h>
#include "mifare.c"
#include "nfc-utils.h"
#include "nfc-utils.c"

// Crypto1
#include "crapto1.h"
#include "crapto1.c"
#include "crypto1.c"

#define MAX_FRAME_LEN 264

// GLOBAL
static uint8_t abtRx[MAX_FRAME_LEN];
static uint8_t abtRxPar[MAX_FRAME_LEN];
static size_t szRx    = sizeof(abtRx);
static size_t szRxPar = sizeof(abtRxPar);
static int szRxBits;
static int szRxParBits;
static nfc_device *pnd;
static nfc_target nt;

bool quiet_output = false;
bool timed = false;

// ISO14443A Commands
uint8_t abtRats[4] =      {0xe0, 0x80, 0x00, 0x00};                                    // last two for crc
uint8_t abtReqa[1] =      {0x26};
uint8_t abtWupa[1] =      {0x52};
uint8_t abtBadShort[1] =  {0x33};
uint8_t abtBad[4] =       {0x29, 0x88, 0x00, 0x00};
uint8_t abtSelectAll[2] = {0x93, 0x20};
uint8_t abtSelectTag[9] = {0x93, 0x70, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00};      // last two bytes for crc
uint8_t abtHalt[4] =      {0x50, 0x00, 0x00, 0x00};                                    // last two bytes for crc

// ISO-DEP Commands
uint8_t abtNack[3] =      {0xb2, 0x00, 0x00};     // last two bytes for crc
uint8_t abtDeselect[3] =  {0xc2, 0x00, 0x00};     // last two bytes for crc

// Mifare Classic
// AUTH --> BLOCK 0x07
// READ --> BLOCK 0x04
// WRITE --> BLOCK 0x04
uint8_t abtClassicAuthA[4] =      {0x60, 0x07, 0x00, 0x00};                                                                                     // Authenticate Trailer block (0x07) to read block 0x4                          // AUTH[1] , ADDR[1], CRC[2]
uint8_t abtClassicAuthB[4] =      {0x61, 0x07, 0x00, 0x00};                                                                                     // Authenticate Trailer block (0x07) to read block 0x4                          // AUTH[1] , ADDR[1], CRC[2]
uint8_t abtClassicRead[4] =       {0x30, 0x04, 0x00, 0x00};                                                                                     // Start from block 4                                                           // READ[1] , ADDR[1], CRC[2]
uint8_t abtClassicWrite[4] =      {0xA0, 0x04, 0x00, 0x00};                                                                                     // Start from block 4                                                           // WRITE[1], ADDR[1], CRC[2]
uint8_t abtClassicWriteData[18] = {0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF, 0x00, 0x00}; // DATA[16], CRC[2]

// Encrypted version of Commands
uint8_t abtArEnc[8] =             {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00};
uint8_t abtArEncPar[8] =          {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00};
uint8_t abtClassicReadEnc[4] =    {0x00, 0x00, 0x00, 0x00};
uint8_t abtClassicReadEncPar[4] = {0x00, 0x00, 0x00, 0x00};

// To store result of READ cmd
uint8_t abtReadResultEnc[16] = {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00};
uint8_t abtReadResult[16] =    {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00};
uint8_t abtTrueCardData[16] =  {0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0x77, 0x77, 0x77, 0x77, 0x77, 0x77, 0x77};

// Mifare Classic UID [7e  a2  42  3d]
uint8_t uid_classic[5] = {0x7e, 0xa2, 0x42, 0x3d, 0xa3};

// KEYS
uint8_t keyA[6] = {0xff, 0xff, 0xff, 0xff, 0xff, 0xff};
uint8_t keyB[6] = {0xff, 0xff, 0xff, 0xff, 0xff, 0xff};

// Reader Nonce 
uint8_t Nr[4] = {0x00, 0x00, 0x00, 0x00};

//////////////// FUNCTIONS /////////////////
// works only with 8 bits numbers
static uint8_t
get_parity(uint8_t val)
{
  uint8_t res = 0;
  while (val)
  {
    res ^= val & 1;
    val >>= 1;
  }
  // odd parity
  if (res == 0x00)
    res = 0x01;
  else if (res == 0x01)
    res = 0x00;
  return res;
}

static void
parity_calculate(uint8_t *pbtTxPar, const uint8_t *pbtTx, const size_t szTx)
{
  size_t szPos;
  for (szPos = 0; szPos < szTx; szPos++)
    pbtTxPar[szPos] = get_parity(pbtTx[szPos]);
}

static bool
transmit_bits(FILE *fp, const uint8_t *pbtTx, const size_t szTxBits)
{
  size_t szPos;
  uint32_t cycles = 0;
  uint8_t pbtTxPar[szTxBits / 8];
  bool isStandardFrame = szTxBits >= 8;

  // standard frame parity only is calculated
  if (isStandardFrame)
    parity_calculate(pbtTxPar, pbtTx, szTxBits / 8);

  // Show transmitted command
  if (!quiet_output)
  {
    // printf("Sent bits:     ");
    // print_hex(pbtTx, szTx);
    printf("=> ");
    if (isStandardFrame)
    { // standard frame
      for (szPos = 0; szPos < szTxBits / 8; szPos++)
      {
        printf("%02x ", pbtTx[szPos]);
        fprintf(fp, "%02x ", pbtTx[szPos]);
      }
      printf("- PA: ");
      for (szPos = 0; szPos < szTxBits / 8; szPos++)
      {
        printf("%d ", pbtTxPar[szPos]);
      }
      printf("\n");
      fprintf(fp, "\n");
    }
    else
    { // short frame
      printf("%02x \n", pbtTx[0]);
      fprintf(fp, "%02x \n", pbtTx[0]);
    }
  }
  // Transmit the bit frame command, we don't use the arbitrary parity feature
  if (timed)
  {
    if (isStandardFrame)
    {
      if ((szRxBits = nfc_initiator_transceive_bits_timed(pnd, pbtTx, szTxBits, pbtTxPar, abtRx, sizeof(abtRx), NULL, &cycles)) < 0)
      {
        printf("<= error %d\n", szRxBits);
        fprintf(fp, "\n");
        return false;
      }
    }
    else
    {
      if ((szRxBits = nfc_initiator_transceive_bits_timed(pnd, pbtTx, szTxBits, NULL, abtRx, sizeof(abtRx), NULL, &cycles)) < 0)
      {
        printf("<= error %d\n", szRxBits);
        fprintf(fp, "\n");
        return false;
      }
    }
    if ((!quiet_output) && (szRxBits > 0))
    {
      printf("Response after %u cycles\n", cycles);
    }
  }
  else
  {
    if (isStandardFrame)
    {
      if ((szRxBits = nfc_initiator_transceive_bits(pnd, pbtTx, szTxBits, pbtTxPar, abtRx, sizeof(abtRx), NULL)) < 0)
      {
        printf("<= error %d\n", szRxBits);
        fprintf(fp, "\n");
        return false;
      }
    }
    else
    {
      if ((szRxBits = nfc_initiator_transceive_bits(pnd, pbtTx, szTxBits, NULL, abtRx, sizeof(abtRx), NULL)) < 0)
      {
        printf("<= error %d\n", szRxBits);
        fprintf(fp, "\n");
        return false;
      }
    }
  }
  // Show received answer
  printf("<= ");
  if (!quiet_output && szRxBits >= 8)
  {
    // printf("size of rx (after unwrap) %i\n", szRxBits);
    for (szPos = 0; szPos < szRxBits / 8 - 1; szPos++)
    {
      printf("%02x ", abtRx[szPos]);
      fprintf(fp, "%02x ", abtRx[szPos]);
    }
  }
  printf("\n");
  fprintf(fp, "\n");
  // Succesful transfer
  return true;
}

static bool
transmit_bits_with_arbitrary_parity_bits(FILE *fp, const uint8_t *pbtTx, const size_t szTxBits, const uint8_t *pbtTxPar)
{
  size_t szPos;
  uint32_t cycles = 0;

  // Show transmitted command
  if (!quiet_output)
  {
    // printf("Sent bits:     ");
    // print_hex(pbtTx, szTx);
    printf("=> ");
    for (szPos = 0; szPos < szTxBits / 8; szPos++)
    {
      printf("%02x ", pbtTx[szPos]);
      fprintf(fp, "%02x ", pbtTx[szPos]);
    }
    printf("- PA: ");
    for (szPos = 0; szPos < szTxBits / 8; szPos++)
    {
      printf("%d ", pbtTxPar[szPos]);
    }
    printf("\n");
    fprintf(fp, "\n");
  }
  // Transmit the bit frame command, we don't use the arbitrary parity feature
  if (timed)
  {
    if ((szRxBits = nfc_initiator_transceive_bits_timed(pnd, pbtTx, szTxBits, pbtTxPar, abtRx, sizeof(abtRx), abtRxPar, &cycles)) < 0)
    {
      printf("<= error %d\n", szRxBits);
      fprintf(fp, "\n");
      return false;
    }
  }
  else
  {
    if ((szRxBits = nfc_initiator_transceive_bits(pnd, pbtTx, szTxBits, pbtTxPar, abtRx, sizeof(abtRx), abtRxPar)) < 0)
    {
      printf("<= error %d\n", szRxBits);
      fprintf(fp, "\n");
      return false;
    }
  }
  // Show received answer
  printf("<= ");
  if (!quiet_output && szRxBits >= 8)
  {
    // printf("size of rx (after unwrap) %i\n", szRxBits);
    for (szPos = 0; szPos < szRxBits / 8 - 1; szPos++)
    {
      printf("%02x ", abtRx[szPos]);
      fprintf(fp, "%02x ", abtRx[szPos]);
    }
  }
  printf("\n");
  fprintf(fp, "\n");
  // Succesful transfer
  return true;
}

void num_to_bytes(uint64_t n, uint32_t len, uint8_t *dest)
{
  while (len--)
  {
    dest[len] = (uint8_t)n;
    n >>= 8;
  }
}

long long unsigned int bytes_to_num(uint8_t *src, uint32_t len)
{
  uint64_t num = 0;
  while (len--)
  {
    num = (num << 8) | (*src);
    src++;
  }
  return num;
}

int trailer_block(uint32_t block){
  // Test if we are in the small or big sectors
  return (block < 128) ? ((block + 1) % 4 == 0) : ((block + 1) % 16 == 0);
}

void redText(){
  printf("\033[1;31m");
}

void yellowText(){
  printf("\033[1;33m");
}

void resetText(){
  printf("\033[0m");
}

void greenText(){
  printf("\033[0;32m");
}

//////////////// MAIN /////////////////
int main(int argc, const char *argv[])
{
  nfc_target nt;
  nfc_context *context;

  // Initialize NFC Context
  nfc_init(&context);
  if (context == NULL)
  {
    printf("Unable to init libnfc (malloc)\n");
    exit(EXIT_FAILURE);
  }
  const char *acLibnfcVersion = nfc_version();
  (void)argc;
  printf("%s uses libnfc %s\n", argv[0], acLibnfcVersion);

  // Open NFC Reader
  pnd = nfc_open(context, NULL);
  if (pnd == NULL)
  {
    printf("ERROR: %s", "Unable to open NFC device.");
    nfc_exit(context);
    exit(EXIT_FAILURE);
  }

  // Initializes settings for collisions handling
  if (nfc_initiator_init_collision(pnd) < 0)
  {
    nfc_perror(pnd, "nfc_initiator_collision_init");
    nfc_close(pnd);
    nfc_exit(context);
    exit(EXIT_FAILURE);
  }

  // Print which reader is used
  printf("NFC reader: %s opened\n", nfc_device_get_name(pnd));

  // Field is active, target id is known, sleep for a second to power up card more
  sleep(0.3);

  // Variable parameters retrieved from CMD Line
  size_t flow_repeat_times = 1;
  size_t command_repeat_times = 5;
  printf("argc %i\n\n", argc);

  if (argc > 2)
  { // 3 or more
    assert(1 == sscanf(argv[2], "%zu", &flow_repeat_times));
    assert(1 == sscanf(argv[1], "%zu", &command_repeat_times));
  }
  else if (argc > 1)
  { // 2
    assert(1 == sscanf(argv[1], "%zu", &command_repeat_times));
  }

  // Prepare all non-variable commands by appending the CRC
  iso14443a_crc_append(abtBad, 2);
  iso14443a_crc_append(abtRats, 2);
  iso14443a_crc_append(abtNack, 1);
  iso14443a_crc_append(abtDeselect, 1);
  iso14443a_crc_append(abtHalt, 2);
  iso14443a_crc_append(abtClassicAuthA, 2);
  iso14443a_crc_append(abtClassicAuthB, 2);
  iso14443a_crc_append(abtClassicRead, 2);
  iso14443a_crc_append(abtClassicWrite, 2);
  iso14443a_crc_append(abtClassicWriteData, 16);

  // Creating a file to save the log
  mkdir("../data/classic", 0777);
  mkdir("../data/classic/expectedTxt", 0777);
  mkdir("../data/classic/decryptionKeys", 0777);
  mkdir("../data/classic/elapsedTime", 0777);
  
  FILE *fp;
  fp = fopen("../data/classic/expectedTxt/classic_output.txt", "w+");

  FILE *decryptingKeysFile;
  decryptingKeysFile = fopen("../data/classic/decryptionKeys/classic_output_decryption_keys.txt", "w+");

  // Start Recording Acquisition TIME
  struct timeval begin, end;
  gettimeofday(&begin, 0);

  // Repeate the communication according to the value passe through cmd line
  for (size_t k = 0; k < flow_repeat_times; k++)
  {
    // Print start of communication
    printf("+++++++++++++++++++++++++ Communication #%ld +++++++++++++++++++++++++++\n", k);

    // Send WUPA [0x52]
    // (also REQA works, but WUPA is better, since it also wakes from HALT)
    transmit_bits(fp, abtWupa, 7);

    // Receive ATQA [0x04 0x00]

    // Send SELECT_ALL [0x93 0x20]
    transmit_bits(fp, abtSelectAll, 2 * 8);

    // Receive UID of the tag [0x7e 0xa2 0x42 0x3d 0xa3]

    // Send SELECT + UID [0x93 0x70] [0x7e 0xa2 0x42 0x3d 0xa3]
    memcpy(abtSelectTag + 2, uid_classic, 5);
    iso14443a_crc_append(abtSelectTag, 7);
    transmit_bits(fp, abtSelectTag, 9 * 8);

    // Receive Type of Card [0x08 0xb6 0xdd]

    ///////////////////// AUTHENTICATION ////////////////////////////
    printf("+++ Authentication START +++\n");

    struct Crypto1State *pcs;
    uint32_t Nt;
    int i;

    // Send AUTH0 for Trailer Block 7 [0x60 0x07 0xd1 0x3d]
    transmit_bits(fp, abtClassicAuthA, 4 * 8);

    // Receive NonceTAG [0xNN 0xNN 0xNN 0xNN]
    Nt = bytes_to_num(abtRx, 4);

    // Init the cipher with key A {0..47} bits
    pcs = crypto1_create(bytes_to_num(keyA, 6));

    // Load (plain) uid^nt into the cipher {48..79} bits
    crypto1_word(pcs, Nt ^ bytes_to_num(uid_classic, 4), 0);

    // Prepare the Reader Anwer:
    // Nr XOR ks1, Ar XOR ks2
    // Ar = suc2(Nt)

    // Prepare Nr XOR ks1
    for (i = 0; i < 4; i++)
    {
      abtArEnc[i] = crypto1_byte(pcs, Nr[i], 0) ^ Nr[i];
      abtArEncPar[i] = filter(pcs->odd) ^ oddparity(Nr[i]);
    }

    // Skip 32 bits in the pseudo random generator to get suc2(Nt)
    Nt = prng_successor(Nt, 32);

    // Prepare Ar XOR ks2
    for (i = 4; i < 8; i++)
    {
      // Get the next random byte
      Nt = prng_successor(Nt, 8);
      // Encrypt the reader-answer (Nt' = suc2(Nt))
      abtArEnc[i] = crypto1_byte(pcs, 0x00, 0) ^ (Nt & 0xff);
      abtArEncPar[i] = filter(pcs->odd) ^ oddparity(Nt);
    }

    // Transmit ReaderAnswer
    transmit_bits_with_arbitrary_parity_bits(fp, abtArEnc, 8 * 8, abtArEncPar);

    // Decrypt the tag answer and verify that suc3(Nt) is At
    Nt = prng_successor(Nt, 32);

    // This check do not work with blocking card so we don't exit
    if (!((crypto1_word(pcs, 0x00, 0) ^ bytes_to_num(abtRx, 4)) == (Nt & 0xFFFFFFFF)))
    {
      //ERR("[At] is not Suc3(Nt), something is wrong, exiting..");
      //exit(EXIT_FAILURE);
    }
    

    printf("+++ Authentication END +++\n");
    /////////////////////////////////////////////////////////////////

    ///////////////////         READ           //////////////////////
    // Send READ for block 4 [0x30 0x04 0x26 0xee]
    // Encrypt READ
    for (i = 0; i < 4; i++)
    {
      abtClassicReadEnc[i] = crypto1_byte(pcs, 0x00, 0) ^ abtClassicRead[i];
      // Encrypt the parity bits with the 4 plaintext bytes
      abtClassicReadEncPar[i] = filter(pcs->odd) ^ oddparity(abtClassicRead[i]);
    }

    // Send Encrypted READ
    transmit_bits_with_arbitrary_parity_bits(fp, abtClassicReadEnc, 4 * 8, abtClassicReadEncPar);

    // Copy result of READ
    memcpy(abtReadResultEnc, abtRx, 16);

    // Ensure card will be in the IDLE state by sending HALT
    // Send HALT [0x50 0x00]
    transmit_bits(fp, abtHalt, 4 * 8);

    /////////////////          DECRYPTION        ///////////////////////////
    // Decrypt answer and store the key used for the dcrypting
    uint8_t decryptingKey[16] = {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00};

    for (i = 0; i < 16; i++)
    { decryptingKey[i] = crypto1_byte(pcs, 0x00, 0);
      abtReadResult[i] = decryptingKey[i] ^ abtReadResultEnc[i];
    }

    yellowText();
    printf("--- RESULTS ---\n");
    printf("--- Decrypting Key:         ");
    for (int szPos = 0; szPos < 128/8; szPos++)
    {
      printf("%02x ", decryptingKey[szPos]);
      fprintf(decryptingKeysFile,"%02x ", decryptingKey[szPos]);
    }
    printf("\n");
    fprintf(decryptingKeysFile, "\n");

    printf("--- Decripted Card Content: ");
    for (int szPos = 0; szPos < 128/8; szPos++)
    {
      printf("%02x ", abtReadResult[szPos]);
    }
    printf("\n");

    printf("--- Expected Card Content:  ");
    for (int szPos = 0; szPos < 128/8; szPos++)
    {
      printf("%02x ", abtTrueCardData[szPos]);
    }
    printf("\n");

    // If READ result decrypted is different from true card content I print an error
    if(bytes_to_num(abtReadResult,16) != bytes_to_num(abtTrueCardData,16)){
      redText();
      printf("ERROR: Data is not correct\n\n");
      resetText();
    }
    else{
      greenText();
      printf("SUCCESS: Data decrypted correctly\n\n");
      resetText();
    }
  
  }

  // Stop Recording and Store Acquisition TIME
  gettimeofday(&end, 0);
  long seconds = end.tv_sec - begin.tv_sec;
  long microseconds = end.tv_usec - begin.tv_usec;
  double elapsed = seconds + microseconds*1e-6;

  printf("ELAPSED TIME %f \n", elapsed);
  fp = fopen("../data/classic/elapsedTime/classic_time_output.txt", "w+");
  fprintf(fp,"%f \n", elapsed);

  // Done, close everything now
  fclose(fp);
  nfc_close(pnd);
  nfc_exit(context);
  exit(EXIT_SUCCESS);
}