import { Component, ElementRef, ViewChild, computed, inject, signal } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { Router } from '@angular/router';

import { AuthService } from '../../services/auth.service';
import { ChatService, QueryResponse, SourceChunk } from '../../services/chat.service';

interface ChatTurn {
  id: number;
  role: 'user' | 'assistant';
  content: string;
  sources?: SourceChunk[];
  accessed_labels?: string[];
  loading?: boolean;
  error?: boolean;
}

@Component({
  selector: 'app-chat',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './chat.component.html',
  styleUrls: ['./chat.component.css'],
})
export class ChatComponent {
  private auth = inject(AuthService);
  private chat = inject(ChatService);
  private router = inject(Router);

  @ViewChild('scroller') scroller?: ElementRef<HTMLDivElement>;

  user = this.auth.user;
  rooms = computed(() => this.user()?.accessible_folders ?? []);
  isAdmin = computed(() => {
    const r = this.user()?.role;
    return r === 'ADMIN' || r === 'CLevel';
  });

  question = '';
  turns = signal<ChatTurn[]>([]);
  asking = signal(false);

  private nextId = 1;

  send() {
    const q = this.question.trim();
    if (!q || this.asking()) return;

    const userTurn: ChatTurn = { id: this.nextId++, role: 'user', content: q };
    const assistantTurn: ChatTurn = {
      id: this.nextId++,
      role: 'assistant',
      content: '',
      loading: true,
    };

    this.turns.update((t) => [...t, userTurn, assistantTurn]);
    this.question = '';
    this.asking.set(true);
    this.scrollSoon();

    this.chat.query(q).subscribe({
      next: (res: QueryResponse) => {
        this.turns.update((t) =>
          t.map((turn) =>
            turn.id === assistantTurn.id
              ? {
                  ...turn,
                  content: res.answer,
                  sources: res.sources,
                  accessed_labels: res.accessed_labels,
                  loading: false,
                }
              : turn
          )
        );
        this.asking.set(false);
        this.scrollSoon();
      },
      error: (err) => {
        const msg =
          err?.error?.detail ?? 'Sorry, the request failed. Please try again.';
        this.turns.update((t) =>
          t.map((turn) =>
            turn.id === assistantTurn.id
              ? { ...turn, content: msg, loading: false, error: true }
              : turn
          )
        );
        this.asking.set(false);
        this.scrollSoon();
      },
    });
  }

  logout() {
    this.auth.logout();
    this.router.navigate(['/login']);
  }

  onEnter(ev: Event) {
    const e = ev as KeyboardEvent;
    if (!e.shiftKey) {
      e.preventDefault();
      this.send();
    }
  }

  trackTurn(_: number, turn: ChatTurn) {
    return turn.id;
  }

  labelTone(label: string): string {
    // Tiny color helper so each data room reads at a glance.
    const map: Record<string, string> = {
      PUBLIC: '#0ea5e9',
      ENGG:   '#22c55e',
      SALES:  '#f59e0b',
      HR:     '#ec4899',
      CLevel: '#8b5cf6',
      INTERN: '#64748b',
    };
    return map[label] ?? '#4f46e5';
  }

  private scrollSoon() {
    queueMicrotask(() => {
      const el = this.scroller?.nativeElement;
      if (el) el.scrollTop = el.scrollHeight;
    });
  }
}
