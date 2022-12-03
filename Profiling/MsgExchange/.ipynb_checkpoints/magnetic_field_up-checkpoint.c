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

#include <nfc/nfc.h>

static nfc_device *pnd;

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

  // field up
  printf("Field up for some seconds.");
  int field_up_seconds = 10;
  if (argc > 1)
    field_up_seconds = atoi(argv[1]);
  sleep(field_up_seconds);

  nfc_close(pnd);
  nfc_exit(context);
  exit(EXIT_SUCCESS);
}
