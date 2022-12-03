// #include <stdlib.h>
// #include <string.h>
// #include <nfc/nfc.h>
// #include <unistd.h>

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif // HAVE_CONFIG_H

#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <unistd.h>
#include <assert.h>

#include <nfc/nfc.h>


#define MAX_FRAME_LEN 264

static uint8_t abtRx[MAX_FRAME_LEN];
static int szRxBits;
static size_t szRx = sizeof(abtRx);

static nfc_device *pnd;


bool    quiet_output = false;
bool    timed = false;

// ISO14443A Commands
// f = max number of frames the PCD can receive 2**8 bytes = 256 bytes
// 1 = PICC addressed by PCD
// 81, 50
uint8_t  abtRats[4] = { 0xe0, 0x80, 0x00, 0x00 }; // last two for crc
uint8_t  abtReqa[1] = { 0x26 };
uint8_t  abtWupa[1] = { 0x52 };
// uint8_t  abtBadShort[1] = { 0x33 };
uint8_t  abtBad[4] = { 0x29, 0x88, 0x00, 0x00 };
uint8_t  abtSelectAll[2] = { 0x93, 0x20 };
uint8_t  abtSelectTag[9] = { 0x93, 0x70, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 }; // last two bytes for crc
uint8_t  abtHalt[4] = { 0x50, 0x00, 0x00, 0x00 }; // last two bytes for crc
uint8_t  abtReadMemory[4] = { 0x30, 0x04, 0x00, 0x00 }; // last two bytes for crc
uint8_t  abtFastRead[5] = { 0x3A, 0x00, 0x13, 0x00, 0x00 }; // last two bytes for crc
uint8_t  abtRead00[4] = { 0x30, 0x00, 0x00, 0x00 }; // last two bytes for crc
// ISO-DEP Commands
uint8_t  abtNack[3] = { 0xb2, 0x00, 0x00 }; // last two bytes for crc
uint8_t  abtDeselect[3] = { 0xc2, 0x00, 0x00 }; // last two bytes for crc
// Mifare classic
uint8_t  abtAuthenticate30[4] = { 0x60, 0x30, 0x00, 0x00 }; // last two bytes for crc

// Mifare Ultralight
uint8_t  abtPwd[7] = { 0x1B, 0x01, 0x02, 0x03, 0x04, 0x00, 0x00 }; // last two bytes for crc
uint8_t  abtWriteData[8] = { 0xA2, 0x04, 0x04, 0x03, 0x02, 0x01, 0x00, 0x00 }; // last two bytes for crc
uint8_t  abtWriteAUTH0[8] = { 0xA2, 0x10, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00 }; // last two bytes for crc
uint8_t  abtWriteAccess[8] = { 0xA2, 0x11, 0x80, 0x05, 0x00, 0x00, 0x00, 0x00 }; // last two bytes for crc
uint8_t  abtWritePwd[8] = { 0xA2, 0x12, 0x01, 0x02, 0x03, 0x04, 0x00, 0x00 }; // last two bytes for crc

// target id is known
const uint8_t uid_selected = 1; // 0=tag1, 1=tag2, 2=mifare classic

// for an ISO/IEC 14443 type A modulation, pbbInitData contains the UID you want to select;
// tag1
// 04 7A A8 A2 36 60 80
const uint8_t tag1_part_1[5] = {0x88, 0x04, 0x7a, 0xa8, 0x5e};
const uint8_t tag1_part_2[5] = {0xa2, 0x36, 0x60, 0x80, 0x74};

// tag2
// 04 FC 0E AA 36 60 84
const uint8_t tag2_part_1[5] = {0x88, 0x04, 0xfc, 0x0e, 0x7e};
const uint8_t tag2_part_2[5] = {0xaa, 0x36, 0x60, 0x84, 0x78};

// dynamic selection of tag1 or tag2
uint8_t uid_part_1[5];
uint8_t uid_part_2[5];

// mifare classic
const uint8_t uid_classic[5] = {0x33, 0x2c, 0x51, 0x24, 0x6a};

// UNUSED
// cc serial
// 87 66 20 9a (bcc = 5b)
// const uint8_t cc_serial[5] = {0x87, 0x66, 0x20, 0x9a, 0x5b}; // cc

// works only with 8 bits numbers
static  uint8_t
get_parity(uint8_t val) {
  uint8_t res = 0;
  while (val) {
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

static  void
parity_calculate(uint8_t *pbtTxPar, const uint8_t *pbtTx, const size_t szTx)
{
  size_t szPos;
  for (szPos = 0; szPos < szTx; szPos++)
    pbtTxPar[szPos] = get_parity(pbtTx[szPos]);
}

static  bool
transmit_bits(FILE *fp, const uint8_t *pbtTx, const size_t szTxBits)
{
  size_t szPos;
  uint32_t cycles = 0;
  uint8_t pbtTxPar[szTxBits/8];
  bool isStandardFrame = szTxBits >= 8;

  // standard frame parity only is calculated
  if (isStandardFrame)
    parity_calculate(pbtTxPar, pbtTx, szTxBits/8);

  // Show transmitted command
  if (!quiet_output) {
    // printf("Sent bits:     ");
    // print_hex(pbtTx, szTx);
    printf("=> ");
    if (isStandardFrame) {    // standard frame
      for (szPos = 0; szPos < szTxBits/8; szPos++) {
        printf("%02x ", pbtTx[szPos]);
        fprintf(fp, "%02x ", pbtTx[szPos]);
      }
      printf("- PA: ", pbtTxPar[szPos]);
      for (szPos = 0; szPos < szTxBits/8; szPos++) {
        printf("%d ", pbtTxPar[szPos]);
      }
      printf("\n");
      fprintf(fp, "\n");
    } else {                // short frame
      printf("%02x \n", pbtTx[0]);
      fprintf(fp, "%02x \n", pbtTx[0]);
    }
  }
  // Transmit the bit frame command, we don't use the arbitrary parity feature
  if (timed) {
    if (isStandardFrame) {
      if ((szRxBits = nfc_initiator_transceive_bits_timed(pnd, pbtTx, szTxBits, pbtTxPar, abtRx, sizeof(abtRx), NULL, &cycles)) < 0) {
        printf("<= error %d\n", szRxBits);
        fprintf(fp, "\n");
        return false;
      }
    }
    else {
      if ((szRxBits = nfc_initiator_transceive_bits_timed(pnd, pbtTx, szTxBits, NULL, abtRx, sizeof(abtRx), NULL, &cycles)) < 0) {
        printf("<= error %d\n", szRxBits);
        fprintf(fp, "\n");
        return false;
      }
    }
    if ((!quiet_output) && (szRxBits > 0)) {
      printf("Response after %u cycles\n", cycles);
    }
  } else {
    if (isStandardFrame) {
      if ((szRxBits = nfc_initiator_transceive_bits(pnd, pbtTx, szTxBits, pbtTxPar, abtRx, sizeof(abtRx), NULL)) < 0) {
        printf("<= error %d\n", szRxBits);
        fprintf(fp, "\n");
        return false;
      }
    }
    else {
      if ((szRxBits = nfc_initiator_transceive_bits(pnd, pbtTx, szTxBits, NULL, abtRx, sizeof(abtRx), NULL)) < 0) {
        printf("<= error %d\n", szRxBits);
        fprintf(fp, "\n");
        return false;
      }
    }
  }
  // Show received answer
  printf("<= ");
  if (!quiet_output && szRxBits >= 8) {
    // printf("size of rx (after unwrap) %i\n", szRxBits);
    for (szPos = 0; szPos < szRxBits/8 - 1; szPos++) {
      printf("%02x ", abtRx[szPos]);
      fprintf(fp, "%02x ", abtRx[szPos]);
    } 
  }
  printf("\n");
  fprintf(fp, "\n");
  // Succesful transfer
  return true;
}

int
main(int argc, const char *argv[])
{
  // nfc_target nt;
  nfc_context *context;
  nfc_init(&context);
  if (context == NULL) {
    printf("Unable to init libnfc (malloc)\n");
    exit(EXIT_FAILURE);
  }
  const char *acLibnfcVersion = nfc_version();
  (void)argc;
  printf("%s uses libnfc %s\n", argv[0], acLibnfcVersion);

  pnd = nfc_open(context, NULL);

  if (pnd == NULL) {
    printf("ERROR: %s", "Unable to open NFC device.");
    nfc_exit(context);
    exit(EXIT_FAILURE);
  }

  // initializes settings for collisions handling
  if (nfc_initiator_init_collision(pnd) < 0) {
    nfc_perror(pnd, "nfc_initiator_collision_init");
    nfc_close(pnd);
    nfc_exit(context);
    exit(EXIT_FAILURE);
  }

  printf("NFC reader: %s opened\n", nfc_device_get_name(pnd));

  // Automatic frames encapsulation is disabled
  // const nfc_modulation nmMifare = {
  //   .nmt = NMT_ISO14443A,
  //   .nbr = NBR_106,
  // };

  // printf("Polling for target...\n");
  // while (nfc_initiator_select_passive_target(pnd, nmMifare, NULL, 0, &nt) <= 0);
  // printf("Target detected!\n");
  
  // res = nfc_initiator_select_passive_target(pnd, nmMifare, uid, sizeof(uid), &nt);
  // printf("nfc_initiator_select_passive_target %i\n", res);

  // field is active, target id is known, sleep for a second to power up card more
  sleep(0.3);

  // variable parameters
  size_t flow_repeat_times = 1;
  size_t command_repeat_times = 5;
  printf("argc %i\n", argc);

  if (argc > 2) { // 3 or more
	  assert (1 == sscanf(argv[2], "%zu", &flow_repeat_times));
	  assert (1 == sscanf(argv[1], "%zu", &command_repeat_times));
  }
  else if (argc > 1) { // 2
	  assert (1 == sscanf(argv[1], "%zu", &command_repeat_times));
  }

  // prepare all non-variable commands by appending the CRC
  iso14443a_crc_append(abtRead00, 2);
  iso14443a_crc_append(abtReadMemory, 2);
  iso14443a_crc_append(abtFastRead, 3);
  iso14443a_crc_append(abtBad, 2);
  iso14443a_crc_append(abtRats, 2);
  iso14443a_crc_append(abtNack, 1);
  iso14443a_crc_append(abtDeselect, 1);
  iso14443a_crc_append(abtAuthenticate30, 2);
  iso14443a_crc_append(abtHalt, 2);
  // mifare ultralight
  iso14443a_crc_append(abtPwd, 5);
  iso14443a_crc_append(abtWriteData, 6);
  iso14443a_crc_append(abtWriteAUTH0, 6);
  iso14443a_crc_append(abtWriteAccess, 6);
  iso14443a_crc_append(abtWritePwd, 6);

  if (uid_selected == 0) {
    memcpy(uid_part_1, tag1_part_1, 5);
    memcpy(uid_part_2, tag1_part_2, 5);
  }
  else {
    memcpy(uid_part_1, tag2_part_1, 5);
    memcpy(uid_part_2, tag2_part_2, 5);
  }

  // creating a file to save the log
  FILE *fp;
  fp = fopen("data.txt", "w+");

  for (size_t k = 0; k < flow_repeat_times; k++) {
    //mifare classic: selection
    if (uid_selected == 2) {
      // 0x52
	    transmit_bits(fp, abtWupa, 7); // or REQA, but WUPA is better, it also wakes from HALT
      
      // 0x93 0x20
      transmit_bits(fp, abtSelectAll, 2*8);
      
      // 0x93 0x70
      memcpy(abtSelectTag + 2, uid_classic, 5);
      iso14443a_crc_append(abtSelectTag, 7);
      transmit_bits(fp, abtSelectTag, 9*8);

      // key A authentication
      // transmit_bits(fp, abtAuthenticate30, 4*8);
      // random nonce as a response!

      // see 3 pass authentication on specific sector (mf-classic) ...


      // Ensure card will be in the IDLE state by sending a wrong command
      // if (k != flow_repeat_times - 1)
        // transmit_bits(fp, abtBad, 4*8); // bad unexpected command
      // else
        // transmit_bits(fp, abtHalt, 4*8); // at the end HALT
      
      // Ensure card will be in the IDLE state by sending HALT
      // 0x50 0x00
      transmit_bits(fp, abtHalt, 4*8);
    }
    // mifare ultralight ev1 selection (no password)
    else if (uid_selected == 1 || uid_selected == 0) {
      // 0x52
	    transmit_bits(fp, abtWupa, 7); // or REQA, but WUPA is better, it also wakes from HALT
      
      // trick: skip anti-collision procedure with Read 0x00
      transmit_bits(fp, abtRead00, 4*8);

      // // TO REMOVE UID KNOWN ASSUMPTION
      // // 0x93 0x20 (gives uid_part_1 as response, save it and use it in the next command)
      // // Prepare CL1 commands
      // // abtSelectAll[0] = 0x93;
      // // Anti-collision
      // // transmit_bits(fp, abtSelectAll, 2*8);

      // // l:Prepare and send CL1 Select-Command
      // // 0x93 0x70 uid_part_1
      // abtSelectTag[0] = 0x93;
      // memcpy(abtSelectTag + 2, uid_part_1, 5);
      // iso14443a_crc_append(abtSelectTag, 7);
      // transmit_bits(fp, abtSelectTag, 9*8);
      // // Ended CL1

      // // TO REMOVE UID KNOWN ASSUMPTION
      // // 0x95 0x20 (gives uid_part_2 as response, save it and use it in the next command)
      // // Prepare CL2 commands
      // // abtSelectAll[0] = 0x95;
      // // Anti-collision
      // // transmit_bits(fp, abtSelectAll, 2*8);

      // // // Check answer
      // // if ((abtRx[0] ^ abtRx[1] ^ abtRx[2] ^ abtRx[3] ^ abtRx[4]) != 0) {
      // //   printf("WARNING: BCC check failed!\n");
      // // }

      // // 0x95 0x70 uid_part_2
      // // l:Selection
      // abtSelectTag[0] = 0x95;
      // memcpy(abtSelectTag + 2, uid_part_2, 5);
      // iso14443a_crc_append(abtSelectTag, 7);
      // transmit_bits(fp, abtSelectTag, 9*8);

      // 0x3A 0x00 0x13
      transmit_bits(fp, abtFastRead, 5*8); // READ from page address 0x00 to 0x13

      // Ensure card will be in the IDLE state by sending a wrong command
      // if (k != flow_repeat_times - 1)
      //   transmit_bits(fp, abtBad, 4*8); // bad unexpected command
      // else
      //   transmit_bits(fp, abtHalt, 4*8); // at the end HALT

      // Ensure card will be in the IDLE state by sending HALT
      // 0x50 0x00
      transmit_bits(fp, abtHalt, 4*8);
    }
    // mifare ultralight read with password
    else if (uid_selected == 3) {
      // 0x52
	    transmit_bits(fp, abtWupa, 7); // or REQA, but WUPA is better, it also wakes from HALT
      
      // trick: skip anti-collision procedure with Read 0x00
      transmit_bits(fp, abtRead00, 4*8);

      // login with password
      transmit_bits(fp, abtPwd, 7*8); // pwd known by reader (can be sniffed from reader)
      // ref. https://ora.ox.ac.uk/objects/uuid:8e52bcfe-5ab5-40b8-b1f4-6b11fd0e67f2
      // https://ieeexplore.ieee.org/abstract/document/7945583/

      // 0x3A 0x00 0x13
      transmit_bits(fp, abtFastRead, 5*8); // READ from page address 0x00 to 0x13

      // Ensure card will be in the IDLE state by sending HALT
      // 0x50 0x00
      transmit_bits(fp, abtHalt, 4*8);
    }
    // program mifare ultralight ev1
    else if (uid_selected == 4) {
      // 0x52
	    transmit_bits(fp, abtWupa, 7); // or REQA, but WUPA is better, it also wakes from HALT
      
      // trick: skip anti-collision procedure with Read 0x00
      transmit_bits(fp, abtRead00, 4*8);

      // write data on memory
      transmit_bits(fp, abtWriteData, 8*8);
      // page address from which verification is required
      transmit_bits(fp, abtWriteAUTH0, 8*8);
      // write access on memory pages
      transmit_bits(fp, abtWriteAccess, 8*8);
      // write password
      transmit_bits(fp, abtWritePwd, 8*8);

      // login with password
      transmit_bits(fp, abtPwd, 7*8);

      // 0x3A 0x00 0x13
      transmit_bits(fp, abtFastRead, 5*8); // READ from page address 0x00 to 0x13

      // Ensure card will be in the IDLE state by sending HALT
      // 0x50 0x00
      transmit_bits(fp, abtHalt, 4*8);
    }

    // more commands
    // Send the RATS command to make a PN532 come out of auto-emulation or to make a card send response
    // transmit_bits(fp, abtRats, 4*8); // not in mf_classic

    // ISO-DEP commands
    // for (size_t i = 0; i < command_repeat_times; i++)
    //   transmit_bits(fp, abtNack, 3*8);
    // transmit_bits(fp, abtDeselect, 3*8);
  }

  // Done, close everything now
  // fprintf(fp, "\n"); // TODO: error, should be removed

  fclose(fp);
  nfc_close(pnd);
  nfc_exit(context);
  exit(EXIT_SUCCESS);
}
