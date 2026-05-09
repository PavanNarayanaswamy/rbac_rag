import { Injectable, computed, signal } from '@angular/core';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import { Observable, tap } from 'rxjs';

export interface UserPublic {
  username: string;
  role: string;
  full_name?: string | null;
  accessible_folders: string[];
}

export interface LoginResponse {
  access_token: string;
  token_type: string;
  user: UserPublic;
}

const TOKEN_KEY = 'rbac_rag_token';
const USER_KEY = 'rbac_rag_user';

@Injectable({ providedIn: 'root' })
export class AuthService {
  /** Reactive signal that holds the currently logged-in user (or null). */
  readonly user = signal<UserPublic | null>(this.loadUser());
  readonly isAuthenticated = computed(() => this.user() !== null);

  // Default backend; override via environment if you containerize.
  private readonly apiBase = 'http://localhost:8000';

  constructor(private http: HttpClient) {}

  login(username: string, password: string): Observable<LoginResponse> {
    // FastAPI's OAuth2PasswordRequestForm expects x-www-form-urlencoded.
    const body = new URLSearchParams();
    body.set('username', username);
    body.set('password', password);

    return this.http
      .post<LoginResponse>(`${this.apiBase}/api/auth/login`, body.toString(), {
        headers: new HttpHeaders({
          'Content-Type': 'application/x-www-form-urlencoded',
        }),
      })
      .pipe(
        tap((res) => {
          localStorage.setItem(TOKEN_KEY, res.access_token);
          localStorage.setItem(USER_KEY, JSON.stringify(res.user));
          this.user.set(res.user);
        })
      );
  }

  logout(): void {
    localStorage.removeItem(TOKEN_KEY);
    localStorage.removeItem(USER_KEY);
    this.user.set(null);
  }

  getToken(): string | null {
    return localStorage.getItem(TOKEN_KEY);
  }

  refreshMe(): Observable<UserPublic> {
    return this.http
      .get<UserPublic>(`${this.apiBase}/api/auth/me`)
      .pipe(
        tap((u) => {
          localStorage.setItem(USER_KEY, JSON.stringify(u));
          this.user.set(u);
        })
      );
  }

  apiUrl(path: string): string {
    return `${this.apiBase}${path}`;
  }

  private loadUser(): UserPublic | null {
    try {
      const raw = localStorage.getItem(USER_KEY);
      return raw ? (JSON.parse(raw) as UserPublic) : null;
    } catch {
      return null;
    }
  }
}
