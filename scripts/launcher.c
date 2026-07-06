/* WhisperFlow-Local launcher shim.
 *
 * A real Mach-O main executable so macOS TCC attributes the Microphone /
 * Accessibility / Input Monitoring permissions to "WhisperFlow-Local"
 * (this bundle) instead of "python3.11". Python runs as a child process
 * and inherits this app's permission identity; signals are forwarded so
 * Quit terminates the whole tree.
 *
 * Compiled by scripts/make_app.sh with -DWFL_REPO / -DWFL_PY.
 */
#include <fcntl.h>
#include <signal.h>
#include <spawn.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/wait.h>
#include <unistd.h>

extern char **environ;
static pid_t child = 0;

static void forward_signal(int sig) {
    if (child > 0) kill(child, sig);
}

int main(void) {
    if (chdir(WFL_REPO) != 0) {
        perror("chdir");
        return 1;
    }
    setenv("WFL_LAUNCHED_FROM_APP", "1", 1);

    char *argv[] = {WFL_PY, "-m", "whisperflow_local", NULL};
    signal(SIGTERM, forward_signal);
    signal(SIGINT, forward_signal);
    signal(SIGHUP, forward_signal);

    /* Capture the child's stdout/stderr (LS-launched apps have no tty). */
    posix_spawn_file_actions_t fa;
    posix_spawn_file_actions_init(&fa);
    char logpath[1024];
    const char *home = getenv("HOME");
    if (home != NULL) {
        snprintf(logpath, sizeof logpath,
                 "%s/Library/Application Support/WhisperFlow-Local/launcher-io.log",
                 home);
        posix_spawn_file_actions_addopen(&fa, 1, logpath,
                                         O_CREAT | O_WRONLY | O_TRUNC, 0644);
        posix_spawn_file_actions_adddup2(&fa, 1, 2);
    }

    int rc = posix_spawn(&child, WFL_PY, &fa, NULL, argv, environ);
    if (rc != 0) {
        fprintf(stderr, "posix_spawn failed: %d\n", rc);
        return 1;
    }
    int status = 0;
    while (waitpid(child, &status, 0) < 0) {
        /* interrupted by a forwarded signal: keep waiting */
    }
    return WIFEXITED(status) ? WEXITSTATUS(status) : 1;
}
